"""
analysis_final_utils.py
=======================

Reusable utilities for the endodontic survival-analysis notebook
(`analysis_final.ipynb`).

This module was extracted from the notebook **conservatively**: it
preserves all analytical logic (cohort assignment, episode construction,
coronal restoration window, statistical tests, Cox model settings, PROBE
audit logic). Where the notebook had repeated or inline helpers, they
have been consolidated here with docstrings and (where practical) type
hints. No analytical defaults were changed.

Critical preserved values
-------------------------
* ``STUDY_END = 2021-01-01`` -- data are locked at this date by the
  upstream SQL query (``DueDate <= '2021-01-01'``). Do **not** change.
* ``WINDOW_DAYS = 365`` -- post-start window for coronal-restoration
  indicators.
* Hebrew text-classification regexes for ``סיווג``.
* Cox model: ``penalizer=0.1``, ``min_positive=20``,
  cluster-robust SE on ``Patient_ID``, PH-assumption check.
* PROBE audit ``EXPECTED_COUNTS`` (manuscript reference).

The notebook should ``import analysis_final_utils as utils`` and call
these helpers; nothing in this module relies on notebook globals.
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import os
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test


# =============================================================================
# Constants and configuration
# =============================================================================

# --- Survival/episode column names (single source of truth) ------------------
ID_COL: str = "Patient_ID"
TIME_COL: str = "duration_days"
EVENT_COL: str = "event"
COHORT_COL: str = "Cohort"

# --- Critical study parameters ----------------------------------------------
# Censoring date. Data are locked at 2021-01-01 by the upstream SQL.
# DO NOT CHANGE without re-running the SQL extract.
STUDY_END: pd.Timestamp = pd.Timestamp("2021-01-01")

# Post-start window for coronal restoration indicators.
WINDOW_DAYS: int = 365

# --- Cohort and restoration display orders ----------------------------------
COHORT_ORDER: List[str] = [
    "Root canal treatment",
    "Root canal retreatment",
    "Apicoectomy",
]

RESTO_ORDER: List[str] = ["Neither", "Sealing only", "Sealing + Crown"]

# --- Hebrew text patterns (preserve EXACTLY) --------------------------------
# Sealing / build-up patterns; crown patterns. Used on the raw 'סיווג' column.
SEALING_PAT: str = r"איטום|מבנה"
CROWN_PAT: str = r"הכתר|כתר"

# --- Manuscript-reference expected counts (PROBE audit) ---------------------
# These are the manuscript reference values; the PROBE audit warns if the
# rebuilt cohort differs. Do NOT silently change these.
EXPECTED_COUNTS: Dict[str, int] = {
    "episodes": 119762,
    "patients": 87185,
    "failures": 3490,
    "censored": 116272,
}

EXPECTED_RESTO_GROUPS: List[str] = ["Neither", "Sealing only", "Sealing + Crown"]

# Requested default model variable list (PROBE complete-case audit fallback).
REQUESTED_DEFAULT_MODEL_VARS: List[str] = [
    "duration_days",
    "event",
    "Coronal_Restoration_Group",
    "AgeGroup",
    "Male",
    "Smoking",
    "Cancer",
    "Diabetes",
    "Hypertension",
    "Biphos_use",
]

# --- Patient-level systemic flags (Table 1 / Cox) ---------------------------
# (output_label -> raw_column) mapping. Hypertension is handled separately
# because the source codes 'no hypertension', so we invert it.
SYSTEMIC_FLAG_MAP: Dict[str, str] = {
    "Smoking": "Smok_No",
    "Cancer": "Cancer_No",
    "Diabetes": "Diabet_No",
    "Biphos_use": "Biphos",
    "Pregnancy": "Pregnancy_No",
    "Allergy": "Allergy_Yes",
    "HeartDisease": "Heart_no",
}
SYSTEMIC_CANDIDATE_COLS: List[str] = [
    "Smoking",
    "Cancer",
    "Diabetes",
    "Biphos_use",
    "Pregnancy",
    "Allergy",
    "HeartDisease",
    "Hypertension",
]

# --- Forest-plot label map (cell 42) ----------------------------------------
FOREST_LABEL_MAP: Dict[str, str] = {
    "AgeGroup_≥60": "Age ≥60 years",
    "AgeGroup_<40": "Age <40 years",
    "Male": "Male sex",
    "Diabetes": "Diabetes mellitus",
    "Cancer": "Malignancy",
    "Smoking": "Smoking",
    "Hypertension": "Hypertension",
    "Biphos_use": "Bisphosphonate use",
    "Coronal_Restoration_Group_Sealing only": "Coronal sealing only",
    "Coronal_Restoration_Group_Sealing + Crown": "Coronal sealing + full-coverage crown",
}


# =============================================================================
# Cleaning helpers
# =============================================================================
def to_dt(s: Any) -> pd.Series:
    """Parse a column to datetime, coercing errors to NaT, day-first."""
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def safe_str(s: Any) -> pd.Series:
    """Coerce to string, replacing NA with empty string."""
    return s.fillna("").astype(str)


def is_Y(series: Any) -> pd.Series:
    """Return 1 if the upper-cased trimmed value equals 'Y', else 0."""
    return (safe_str(series).str.strip().str.upper() == "Y").astype(int)


def contains(series: Any, pattern: str) -> pd.Series:
    """Vectorised regex `contains`, NA-safe."""
    return safe_str(series).str.contains(pattern, regex=True, na=False)


# =============================================================================
# Cohort assignment (Hebrew text in 'סיווג')
# =============================================================================
def assign_cohort(text: Any) -> Optional[str]:
    """Map a raw 'סיווג' string into one of the three cohorts or None.

    Rules (preserve EXACTLY):
      * 'אפיס' or 'אפיסקט' or English 'apic' -> 'Apicoectomy'
      * 'חידוש' -> 'Root canal retreatment'
      * 'טיפול שורש' -> 'Root canal treatment'
      * otherwise -> None
    """
    t = str(text)
    tl = t.lower()
    if "אפיס" in t or "אפיסקט" in t or "apic" in tl:
        return "Apicoectomy"
    if "חידוש" in t:
        return "Root canal retreatment"
    if "טיפול שורש" in t:
        return "Root canal treatment"
    return None


# =============================================================================
# Raw cleaning + patient table
# =============================================================================
def clean_raw_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """In-place cleanup of the raw DataFrame: parse dates, coerce IDs,
    add row-level failure flag and Hebrew text indicators.

    Returns the same DataFrame (mutated) for convenience. Adds:
      Treatment_Date_dt, Failure_Treatment_Date_dt,
      First_Initial_Treatment_Date_dt, Treatment_Status_norm,
      failure_flag_row, Cohort, Has_Sealing_row, Has_Crown_row.
    """
    # Date parsing (guarded)
    raw["Treatment_Date_dt"] = to_dt(raw.get("Treatment_Date"))
    raw["Failure_Treatment_Date_dt"] = to_dt(raw.get("Failure_Treatment_Date"))
    raw["First_Initial_Treatment_Date_dt"] = to_dt(raw.get("First_Initial_Treatment_Date"))

    # Core IDs
    raw["Patient_ID"] = pd.to_numeric(raw.get("Patient_ID"), errors="coerce").astype("Int64")
    raw["Tooth_Num"] = pd.to_numeric(raw.get("Tooth_Num"), errors="coerce").astype("Int64")

    # Outcome (row-level) normalisation if present
    raw["Treatment_Status_norm"] = safe_str(raw.get("Treatment_Status")).str.strip().str.lower()
    raw["failure_flag_row"] = (
        raw["Failure_Treatment_Date_dt"].notna()
        | raw.get("failure_code").notna()
        | (raw["Treatment_Status_norm"] == "failure")
    ).astype(int)

    # Cohort assignment from 'סיווג'
    raw["סיווג"] = safe_str(raw["סיווג"])
    raw["Cohort"] = raw["סיווג"].apply(assign_cohort)

    # Row-level coronal restoration indicators
    raw["Has_Sealing_row"] = contains(raw["סיווג"], SEALING_PAT).astype(int)
    raw["Has_Crown_row"] = contains(raw["סיווג"], CROWN_PAT).astype(int)

    return raw


def build_patient_table(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build patient-level covariate table (one row per Patient_ID).

    Mirrors notebook cell 10 exactly:
      * Take first record per patient (sorted by Treatment_Date_dt) for
        Mac_Gender, Age_In_Treatment.
      * Sex coding: 'ז' -> Male=1, 'נ' -> Male=0.
      * AgeGroup with a 40-60 reference category.
      * Systemic flags: Y/N strings via :func:`is_Y`. Hypertension is
        derived from ``Hipertonia_Yes`` (the 'No' column) by inversion.
      * Aggregated by Patient_ID with max() over rows.

    Returns
    -------
    patient : DataFrame
    systemic_cols : list of systemic column names that ended up in the
        patient table (the order matches :data:`SYSTEMIC_CANDIDATE_COLS`).
    """
    patient_cols = ["Patient_ID", "Mac_Gender", "Age_In_Treatment"]
    keep_cols = [c for c in patient_cols if c in raw.columns]

    patient_base = (
        raw.dropna(subset=["Patient_ID"])
        .sort_values("Treatment_Date_dt")
        .groupby("Patient_ID", as_index=False)
        .agg({c: "first" for c in keep_cols})
    )

    # Sex coding: ז = male, נ = female
    patient_base["Male"] = (
        safe_str(patient_base.get("Mac_Gender")).str.strip() == "ז"
    ).astype(int)

    # Age + age group with 40-60 reference
    patient_base["Age"] = pd.to_numeric(
        patient_base.get("Age_In_Treatment"), errors="coerce"
    )
    patient_base["AgeGroup"] = pd.cut(
        patient_base["Age"],
        bins=[-np.inf, 40, 60, np.inf],
        labels=["<40", "40–60", "≥60"],
    )

    # Systemic flags (guarded)
    present_sys = {k: v for k, v in SYSTEMIC_FLAG_MAP.items() if v in raw.columns}
    patient_sys = (
        raw.dropna(subset=["Patient_ID"])[
            ["Patient_ID"] + list(present_sys.values())
        ].copy()
    )
    for out, col in present_sys.items():
        patient_sys[out] = is_Y(raw[col])

    # Hypertension: Hipertonia_Yes == 'Y' means NO hypertension
    if "Hipertonia_Yes" in raw.columns:
        no_htn = is_Y(raw["Hipertonia_Yes"])
        patient_sys["Hypertension"] = (
            (1 - no_htn).where(raw["Hipertonia_Yes"].notna(), 0).astype(int)
        )
        present_sys["Hypertension"] = "Hipertonia_Yes"  # marker only

    agg_cols = [c for c in list(present_sys.keys()) if c in patient_sys.columns]
    patient_sys = (
        patient_sys[["Patient_ID"] + agg_cols]
        .groupby("Patient_ID", as_index=False)
        .max()
    )

    patient = patient_base.merge(patient_sys, on="Patient_ID", how="left")
    systemic_cols = [c for c in SYSTEMIC_CANDIDATE_COLS if c in patient.columns]
    return patient, systemic_cols


# =============================================================================
# Episode construction
# =============================================================================
def build_episodes(raw: pd.DataFrame, study_end: pd.Timestamp = STUDY_END) -> pd.DataFrame:
    """Build one episode per Patient_ID × Tooth_Num × Cohort.

    Logic (preserve EXACTLY -- mirrors notebook cell 11):
      * start_date = min(Treatment_Date_dt) within group.
      * failure_date = min(Failure_Treatment_Date_dt) within group, where
        Failure_Treatment_Date_dt is non-null and >= start_date.
      * event = 1 if failure_date is non-null else 0.
      * stop_date = failure_date if event else ``study_end``.
      * duration_days = stop_date - start_date in whole days.
      * episode_id = "Patient_ID|Tooth_Num|Cohort|YYYY-MM-DD".
      * Drop rows with missing duration_days or duration < 0.
    """
    cohort_rows = raw.dropna(
        subset=["Patient_ID", "Tooth_Num", "Treatment_Date_dt", "Cohort"]
    ).copy()

    starts = (
        cohort_rows.groupby(["Patient_ID", "Tooth_Num", "Cohort"], as_index=False)
        .agg(start_date=("Treatment_Date_dt", "min"))
    )

    tmp = cohort_rows.merge(
        starts, on=["Patient_ID", "Tooth_Num", "Cohort"], how="inner"
    )
    tmp = tmp[tmp["Failure_Treatment_Date_dt"].notna()].copy()
    tmp = tmp[tmp["Failure_Treatment_Date_dt"] >= tmp["start_date"]].copy()

    min_fail = (
        tmp.groupby(["Patient_ID", "Tooth_Num", "Cohort"], as_index=False)
        .agg(failure_date=("Failure_Treatment_Date_dt", "min"))
    )

    episodes = starts.merge(
        min_fail, on=["Patient_ID", "Tooth_Num", "Cohort"], how="left"
    )
    episodes["event"] = episodes["failure_date"].notna().astype(int)
    episodes["stop_date"] = pd.to_datetime(
        episodes["failure_date"].fillna(study_end), errors="coerce"
    )
    episodes["duration_days"] = (
        episodes["stop_date"] - pd.to_datetime(episodes["start_date"], errors="coerce")
    ).dt.days

    episodes["episode_id"] = (
        episodes["Patient_ID"].astype(str)
        + "|"
        + episodes["Tooth_Num"].astype(str)
        + "|"
        + episodes["Cohort"].astype(str)
        + "|"
        + pd.to_datetime(episodes["start_date"]).dt.strftime("%Y-%m-%d")
    )

    episodes = episodes.dropna(subset=["duration_days"]).copy()
    episodes = episodes[episodes["duration_days"] >= 0].copy()
    return episodes


# =============================================================================
# Coronal restoration: episode-level flags within window
# =============================================================================
def add_coronal_restoration_flags(
    episodes: pd.DataFrame,
    raw: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Add ``Has_Sealing_in_window``, ``Has_Crown_in_window`` and
    ``Coronal_Restoration_Group`` to ``episodes`` based on row-level flags
    in ``raw`` falling inside [start_date, min(start_date+window, stop_date)].

    Mirrors notebook cell 12 exactly. The ``window_days`` is exposed as a
    parameter and defaults to :data:`WINDOW_DAYS` (365).
    """
    episodes = episodes.copy()
    episodes["start_date"] = pd.to_datetime(episodes["start_date"], errors="coerce")
    episodes["stop_date"] = pd.to_datetime(episodes["stop_date"], errors="coerce")

    episodes["_window_end"] = episodes["start_date"] + pd.Timedelta(days=window_days)
    episodes["_window_end"] = episodes[["_window_end", "stop_date"]].min(axis=1)

    tmp = episodes[
        ["episode_id", "Patient_ID", "Tooth_Num", "start_date", "_window_end"]
    ].merge(
        raw[
            [
                "Patient_ID",
                "Tooth_Num",
                "Treatment_Date_dt",
                "Has_Sealing_row",
                "Has_Crown_row",
            ]
        ],
        on=["Patient_ID", "Tooth_Num"],
        how="left",
    )

    in_window = (
        tmp["Treatment_Date_dt"].notna()
        & tmp["start_date"].notna()
        & tmp["_window_end"].notna()
        & (tmp["Treatment_Date_dt"] >= tmp["start_date"])
        & (tmp["Treatment_Date_dt"] <= tmp["_window_end"])
    )

    tmpw = tmp.loc[in_window].copy()

    resto_ep = (
        tmpw.groupby("episode_id", as_index=False)
        .agg(
            Has_Sealing_in_window=("Has_Sealing_row", "max"),
            Has_Crown_in_window=("Has_Crown_row", "max"),
        )
    )

    episodes = episodes.drop(
        columns=["Has_Sealing_in_window", "Has_Crown_in_window"], errors="ignore"
    )
    episodes = episodes.merge(resto_ep, on="episode_id", how="left")
    episodes["Has_Sealing_in_window"] = (
        episodes["Has_Sealing_in_window"].fillna(0).astype(int)
    )
    episodes["Has_Crown_in_window"] = (
        episodes["Has_Crown_in_window"].fillna(0).astype(int)
    )

    # Group label (used for plots/tests/models)
    episodes["Coronal_Restoration_Group"] = "Neither"
    episodes.loc[
        (episodes["Has_Sealing_in_window"] == 1)
        & (episodes["Has_Crown_in_window"] == 0),
        "Coronal_Restoration_Group",
    ] = "Sealing only"
    episodes.loc[
        (episodes["Has_Crown_in_window"] == 1),
        "Coronal_Restoration_Group",
    ] = "Sealing + Crown"

    episodes = episodes.drop(columns=["_window_end"], errors="ignore")
    return episodes


def merge_patient_into_episodes(
    episodes: pd.DataFrame, patient: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge patient covariates into episodes on Patient_ID,
    dropping ``Mac_Gender`` and ``Age_In_Treatment`` from the patient
    table first (mirrors notebook cell 13 exactly).
    """
    drop_cols = [
        c for c in ["Mac_Gender", "Age_In_Treatment"] if c in patient.columns
    ]
    return episodes.merge(
        patient.drop(columns=drop_cols), on="Patient_ID", how="left"
    )


# =============================================================================
# Variable inventory
# =============================================================================
def _binary_01(s: pd.Series) -> bool:
    """Return True if the (numeric-coerced) series only contains 0/1."""
    s2 = pd.to_numeric(s, errors="coerce")
    u = set(s2.dropna().unique().tolist())
    return len(u) >= 1 and u.issubset({0, 1})


def build_variable_lists(
    df: pd.DataFrame,
    cox_covariates: Optional[Sequence[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    cohort_col: str = COHORT_COL,
) -> Dict[str, List[str]]:
    """Compute the canonical variable lists used downstream.

    Mirrors notebook cell 14 exactly.

    Returns a dict with keys:
      * ``cox_covariates`` - explicit list (if provided), else all binary
        0/1 columns excluding survival columns.
      * ``categorical_vars`` - object/categorical/bool columns plus
        ``cohort_col``; falls back to ``cox_covariates`` if none found.
      * ``binary_vars`` - all 0/1 columns (excluding survival columns).
    """
    if exclude is None:
        exclude = set()
    exclude = set(exclude) | {id_col, time_col, event_col}

    # 1) Cox covariates (as given) OR fallback to 0/1 dummies + key categoricals
    if cox_covariates is not None:
        cox_list = [c for c in cox_covariates if c in df.columns and c not in exclude]
    else:
        cox_list = []
        for c in df.columns:
            if c in exclude:
                continue
            if _binary_01(df[c]):
                cox_list.append(c)

    # 2) Categorical vars for chi2 / log-rank
    categorical: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if c == cohort_col:
            categorical.append(c)
            continue
        if (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
        ):
            categorical.append(c)

    # If none found (or you mostly work with dummies), fall back to 0/1 list
    if len(categorical) == 0:
        categorical = [c for c in cox_list if _binary_01(df[c])]

    # 3) Binary/systemic list (for Table-1 style %)
    binary = [c for c in df.columns if c not in exclude and _binary_01(df[c])]

    return {
        "cox_covariates": cox_list,
        "categorical_vars": categorical,
        "binary_vars": binary,
    }


# =============================================================================
# Filters and sensitivity datasets
# =============================================================================
def apply_base_episode_filters(
    episodes: pd.DataFrame,
    cohort_col: str = COHORT_COL,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """Apply the consistent base filtering used across all sections.

    Mirrors the cell-16 'episodes_base' construction:
      * Strip whitespace on ``cohort_col``, replace 'nan' string with NaN.
      * Coerce ``time_col`` to numeric, ``event_col`` to numeric->int (NA->0).
      * Drop rows missing time/event.
      * Keep duration >= 0.
    """
    out = episodes.copy()
    if cohort_col in out.columns:
        out[cohort_col] = (
            out[cohort_col].astype(str).str.strip().replace({"nan": np.nan})
        )
    out[time_col] = pd.to_numeric(out.get(time_col), errors="coerce")
    out[event_col] = (
        pd.to_numeric(out.get(event_col), errors="coerce").fillna(0).astype(int)
    )
    out = out.dropna(subset=[time_col, event_col])
    out = out[out[time_col] >= 0]
    return out


def analysis_counts(
    df: pd.DataFrame,
    label: str,
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> pd.Series:
    """Compact one-row analysis-cohort summary used for sanity reporting."""
    return pd.Series(
        {
            "Label": label,
            "Episodes (n)": int(len(df)),
            "Unique patients (n)": int(df[id_col].nunique()) if id_col in df.columns else np.nan,
            "Failures (n)": int(df[event_col].sum()) if event_col in df.columns else np.nan,
            "Failure rate (%)": float(100 * df[event_col].mean()) if len(df) else np.nan,
            "Median follow-up (days)": float(df[time_col].median()) if time_col in df.columns else np.nan,
        }
    )


def first_episode_per_patient(
    episodes: pd.DataFrame,
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Sensitivity dataset: keep the first (shortest-duration-first sort)
    episode per patient. Mirrors the notebook's ``first_episode_df``."""
    return (
        episodes.sort_values(time_col).groupby(id_col, as_index=False).first()
    )


def landmark_dataset(
    episodes: pd.DataFrame,
    landmark_days: int,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Sensitivity dataset: drop failures before ``landmark_days`` and
    reset the time origin (subtract ``landmark_days``)."""
    out = episodes.copy()
    out = out[out[time_col] > landmark_days].copy()
    out[time_col] = out[time_col] - landmark_days
    return out


# =============================================================================
# Validation helpers (new)
# =============================================================================
def summarize_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return basic shape/duplicate validation info for a DataFrame."""
    return pd.DataFrame(
        {
            "name": [name],
            "rows": [len(df)],
            "columns": [df.shape[1]],
            "duplicate_rows": [int(df.duplicated().sum())],
        }
    )


def summarize_survival_dataset(
    df: pd.DataFrame,
    label: str,
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> pd.Series:
    """Key survival-analysis validation counts for a DataFrame."""
    return pd.Series(
        {
            "label": label,
            "episodes": len(df),
            "unique_patients": df[id_col].nunique() if id_col in df.columns else pd.NA,
            "failures": int(df[event_col].sum()) if event_col in df.columns else pd.NA,
            "censored": int((1 - df[event_col]).sum()) if event_col in df.columns else pd.NA,
            "min_duration": df[time_col].min() if time_col in df.columns else pd.NA,
            "max_duration": df[time_col].max() if time_col in df.columns else pd.NA,
        }
    )


# =============================================================================
# Multiple testing
# =============================================================================
def fdr_bh(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment. NaNs are preserved."""
    p = np.array([np.nan if v is None else v for v in pvals], dtype=float)
    out = np.full_like(p, np.nan)
    m = int(np.sum(~np.isnan(p)))
    if m == 0:
        return out
    idx = np.argsort(p, kind="mergesort")
    ranked = p[idx]
    valid = ~np.isnan(ranked)
    idx_valid = idx[valid]
    ranked_valid = ranked[valid]
    q = ranked_valid * m / (np.arange(1, len(ranked_valid) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out[idx_valid] = q
    return out


# =============================================================================
# Descriptive summaries
# =============================================================================
def summarize_groups(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """Episode-level descriptive summary by ``group_cols``.

    If ``group_cols`` is empty, returns a single-row 'Overall' summary
    (prevents pandas groupby([]) errors).
    """
    d = df.copy()
    d[time_col] = pd.to_numeric(d.get(time_col), errors="coerce")
    d[event_col] = (
        pd.to_numeric(d.get(event_col), errors="coerce").fillna(0).astype(int)
    )

    def _one_group(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "Episodes (n)": len(g),
                "Unique patients (n)": g[id_col].dropna().nunique() if id_col in g.columns else np.nan,
                "Failures (n)": int(g[event_col].sum()) if event_col in g.columns else np.nan,
                "Failure rate (%)": 100.0 * g[event_col].mean() if len(g) and event_col in g.columns else np.nan,
                "Success rate (%)": 100.0 * (1 - g[event_col].mean()) if len(g) and event_col in g.columns else np.nan,
                "Follow-up (days) median": g[time_col].median() if time_col in g.columns else np.nan,
                "Follow-up (days) IQR": (
                    g[time_col].quantile(0.75) - g[time_col].quantile(0.25)
                )
                if time_col in g.columns
                else np.nan,
            }
        )

    if not list(group_cols):
        out = _one_group(d).to_frame().T
        out.insert(0, "Group", "Overall")
        return out

    out = d.groupby(list(group_cols), dropna=False).apply(_one_group).reset_index()
    return out


def table1_population(
    df: pd.DataFrame,
    group_col: str,
    systemic_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compact Table-1 style population description (episodes as rows)."""
    d = df.copy()
    d["Age"] = pd.to_numeric(d.get("Age"), errors="coerce")

    if systemic_cols is None:
        systemic_cols = []
    cols = ["Age", "Male"] + [c for c in systemic_cols if c in d.columns]

    rows = []
    for name, g in d.groupby(group_col, dropna=False):
        r = {
            "Group": name,
            "Episodes (n)": len(g),
            "Patients (n)": g["Patient_ID"].nunique() if "Patient_ID" in g.columns else np.nan,
        }
        r["Age mean (SD)"] = (
            f"{g['Age'].mean():.1f} ({g['Age'].std():.1f})"
            if g["Age"].notna().any()
            else ""
        )
        r["Age median (IQR)"] = (
            f"{g['Age'].median():.1f} ({(g['Age'].quantile(0.75) - g['Age'].quantile(0.25)):.1f})"
            if g["Age"].notna().any()
            else ""
        )
        r["Female (%)"] = (
            f"{100*(1-g['Male'].mean()):.1f}"
            if "Male" in g.columns and g["Male"].notna().any()
            else ""
        )

        for c in [c for c in cols if c not in ["Age", "Male"]]:
            r[f"{c} (%)"] = (
                f"{100*g[c].mean():.1f}"
                if c in g.columns and g[c].notna().any()
                else ""
            )

        rows.append(r)

    return pd.DataFrame(rows)


def outcomes_like_cox(
    df: pd.DataFrame,
    covariate_cols: Sequence[str],
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    include_reference_row: bool = True,
) -> pd.DataFrame:
    """Cox-style descriptive outcomes for binary covariates.

    For each binary 0/1 covariate, reports episode/patient/failure counts
    and median follow-up at level==0 (Reference) and level==1 (Compared).
    """
    d = df.copy()
    d[time_col] = pd.to_numeric(d.get(time_col), errors="coerce")
    d[event_col] = (
        pd.to_numeric(d.get(event_col), errors="coerce").fillna(0).astype(int)
    )

    def _summ(g: pd.DataFrame) -> Dict[str, Any]:
        return {
            "Episodes (n)": int(len(g)),
            "Unique patients (n)": int(g[id_col].nunique()) if id_col in g.columns else np.nan,
            "Failures (n)": int(g[event_col].sum()) if event_col in g.columns else np.nan,
            "Failure rate (%)": float(100.0 * g[event_col].mean()) if len(g) else np.nan,
            "Success rate (%)": float(100.0 * (1 - g[event_col].mean())) if len(g) else np.nan,
            "Follow-up (days) median": float(g[time_col].median()) if time_col in g.columns else np.nan,
            "Follow-up (days) IQR": float(
                g[time_col].quantile(0.75) - g[time_col].quantile(0.25)
            )
            if time_col in g.columns
            else np.nan,
        }

    rows: List[Dict[str, Any]] = []
    base = d.dropna(subset=[event_col])

    for v in covariate_cols:
        if v not in base.columns:
            rows.append({"Variable": v, "Level": None, "note": "missing column in df"})
            continue

        x = pd.to_numeric(base[v], errors="coerce")
        ok = x.isin([0, 1]) & base[event_col].notna()

        g0 = base.loc[ok & (x == 0)]
        g1 = base.loc[ok & (x == 1)]

        if len(g0) == 0 and len(g1) == 0:
            rows.append({"Variable": v, "Level": None, "note": "no valid 0/1 values"})
            continue

        if include_reference_row:
            r0 = {"Variable": v, "Level": "Reference (0)"}
            r0.update(_summ(g0))
            r0["note"] = ""
            rows.append(r0)

        r1 = {"Variable": v, "Level": "Compared (1)"}
        r1.update(_summ(g1))
        r1["note"] = ""
        rows.append(r1)

    out = pd.DataFrame(rows)
    core = [
        "Variable",
        "Level",
        "Episodes (n)",
        "Unique patients (n)",
        "Failures (n)",
        "Failure rate (%)",
        "Success rate (%)",
        "Follow-up (days) median",
        "Follow-up (days) IQR",
        "note",
    ]
    for c in core:
        if c not in out.columns:
            out[c] = ""
    return out[core]


def outcomes_like_cox_within_groups(
    df: pd.DataFrame,
    group_col: str,
    covariate_cols: Sequence[str],
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    include_reference_row: bool = True,
) -> pd.DataFrame:
    """Run :func:`outcomes_like_cox` separately within each level of
    ``group_col`` and stack with the grouping column prepended."""
    d = df.copy()
    d[time_col] = pd.to_numeric(d.get(time_col), errors="coerce")
    d[event_col] = (
        pd.to_numeric(d.get(event_col), errors="coerce").fillna(0).astype(int)
    )

    rows = []
    for grp, dg in d.groupby(group_col, dropna=False):
        t = outcomes_like_cox(
            dg,
            covariate_cols,
            id_col=id_col,
            time_col=time_col,
            event_col=event_col,
            include_reference_row=include_reference_row,
        )
        t.insert(0, group_col, grp)
        rows.append(t)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Detailed Table 1 (cell 25)
# =============================================================================
def classify_tooth(num: Any) -> Tuple[str, str]:
    """FDI two-digit tooth number -> (Arch, Position).

    Mirrors notebook cell 25.
    """
    try:
        n = int(num)
    except (ValueError, TypeError):
        return "Unknown", "Unknown"
    quadrant = n // 10
    tooth_in_quad = n % 10
    arch = "Upper" if quadrant in [1, 2] else ("Lower" if quadrant in [3, 4] else "Unknown")
    position = (
        "Anterior"
        if tooth_in_quad in [1, 2, 3]
        else ("Posterior" if tooth_in_quad in [4, 5, 6, 7, 8] else "Unknown")
    )
    return arch, position


def resolve_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Resolve a column name accounting for ``_x``/``_y`` merge suffixes."""
    for suffix in ["", "_x", "_y"]:
        if name + suffix in df.columns:
            return name + suffix
    return None


def build_detailed_table1(rct: pd.DataFrame) -> pd.DataFrame:
    """Build the detailed Table 1 for the primary RCT cohort
    (reviewer comments #76 + #92). Mirrors notebook cell 25 exactly."""
    rct = rct.copy()
    rct[["Arch", "Position"]] = rct["Tooth_Num"].apply(
        lambda x: pd.Series(classify_tooth(x))
    )

    age_col = resolve_column(rct, "Age")
    male_col = resolve_column(rct, "Male")
    smoke_col = resolve_column(rct, "Smoking")
    cancer_col = resolve_column(rct, "Cancer")
    diab_col = resolve_column(rct, "Diabetes")
    biphos_col = resolve_column(rct, "Biphos_use")
    hypert_col = resolve_column(rct, "Hypertension")
    agegrp_col = resolve_column(rct, "AgeGroup")

    n_total = len(rct)
    n_patients = rct["Patient_ID"].nunique() if "Patient_ID" in rct.columns else float("nan")

    rows: List[Dict[str, Any]] = []
    rows.append({"Characteristic": "Episodes (n)", "n": n_total, "%": ""})
    rows.append({"Characteristic": "Unique patients (n)", "n": int(n_patients), "%": ""})

    # Age
    if age_col:
        a = pd.to_numeric(rct[age_col], errors="coerce")
        rows.append(
            {
                "Characteristic": "Age, mean \u00b1 SD (years)",
                "n": f"{a.mean():.1f} \u00b1 {a.std():.1f}",
                "%": "",
            }
        )
        rows.append(
            {
                "Characteristic": "Age, median (IQR)",
                "n": f"{a.median():.1f} ({a.quantile(0.25):.1f}\u2013{a.quantile(0.75):.1f})",
                "%": "",
            }
        )

    # Age groups
    if agegrp_col:
        for ag in ["<40", "40\u201360", "\u226560"]:
            candidates = [
                v
                for v in rct[agegrp_col].dropna().unique()
                if str(ag) in str(v) or ag == str(v)
            ]
            for ag_val in candidates:
                mask = rct[agegrp_col] == ag_val
                rows.append(
                    {
                        "Characteristic": f"  Age {ag_val}, n (%)",
                        "n": int(mask.sum()),
                        "%": f"{100*mask.mean():.1f}",
                    }
                )

    # Sex
    if male_col:
        female = 1 - pd.to_numeric(rct[male_col], errors="coerce").fillna(0)
        rows.append(
            {
                "Characteristic": "Female, n (%)",
                "n": int(female.sum()),
                "%": f"{100*female.mean():.1f}",
            }
        )
        rows.append(
            {
                "Characteristic": "Male, n (%)",
                "n": int(rct[male_col].sum()),
                "%": f"{100*rct[male_col].mean():.1f}",
            }
        )

    # Systemic conditions
    for label, col in [
        ("Diabetes mellitus, n (%)", diab_col),
        ("Smoking, n (%)", smoke_col),
        ("Malignancy, n (%)", cancer_col),
        ("Hypertension, n (%)", hypert_col),
        ("Bisphosphonate use, n (%)", biphos_col),
    ]:
        if col:
            s = pd.to_numeric(rct[col], errors="coerce").fillna(0)
            rows.append(
                {
                    "Characteristic": label,
                    "n": int(s.sum()),
                    "%": f"{100*s.mean():.1f}",
                }
            )

    # Coronal restoration groups
    rows.append({"Characteristic": "--- Restoration ---", "n": "", "%": ""})
    for rg in ["Sealing + Crown", "Sealing only", "Neither"]:
        mask = rct["Coronal_Restoration_Group"] == rg
        rows.append(
            {
                "Characteristic": f"  {rg}, n (%)",
                "n": int(mask.sum()),
                "%": f"{100*mask.mean():.1f}",
            }
        )

    # Tooth location
    rows.append({"Characteristic": "--- Tooth location ---", "n": "", "%": ""})
    for arch in ["Upper", "Lower", "Unknown"]:
        mask = rct["Arch"] == arch
        if mask.sum() > 0:
            rows.append(
                {
                    "Characteristic": f"  Arch: {arch}, n (%)",
                    "n": int(mask.sum()),
                    "%": f"{100*mask.mean():.1f}",
                }
            )
    for pos in ["Anterior", "Posterior", "Unknown"]:
        mask = rct["Position"] == pos
        if mask.sum() > 0:
            rows.append(
                {
                    "Characteristic": f"  Position: {pos}, n (%)",
                    "n": int(mask.sum()),
                    "%": f"{100*mask.mean():.1f}",
                }
            )

    # Outcomes
    rows.append({"Characteristic": "--- Outcomes ---", "n": "", "%": ""})
    n_fail = int(rct["event"].sum())
    rows.append(
        {
            "Characteristic": "Failures (extractions), n (%)",
            "n": n_fail,
            "%": f"{100*rct['event'].mean():.2f}",
        }
    )
    rows.append(
        {
            "Characteristic": "Survival (censored), n (%)",
            "n": n_total - n_fail,
            "%": f"{100*(1-rct['event'].mean()):.2f}",
        }
    )

    table1_detailed = pd.DataFrame(rows)[["Characteristic", "n", "%"]]
    table1_detailed.columns = [
        "Characteristic",
        "n  (or mean \u00b1 SD)",
        "% (or median IQR)",
    ]
    return table1_detailed


# =============================================================================
# Chi-square / Fisher / log-rank batches
# =============================================================================
# NOTE on duplicate definitions in the original notebook:
# Cell 27 defines `chi2_or_fisher_2way` WITHOUT odds-ratio for the chi-square
# 2x2 branch. Cell 29 redefines it WITH odds-ratio (Haldane-Anscombe corrected
# when any zero). Because cell 29 runs after cell 27 and `run_chi2_batch`
# resolves `chi2_or_fisher_2way` at call time (not at definition time), the
# RUNTIME behavior of every chi2/fisher result in the original notebook is
# the cell-29 version. We keep ONLY the cell-29 version here; this preserves
# the original outputs.
def chi2_or_fisher_2way(
    df: pd.DataFrame,
    row: str,
    col: str,
    min_expected: float = 5,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Two-way chi-square / Fisher exact test with odds ratio for 2x2.

    Behaviour (preserved from notebook cell 29):
      * If contingency shape < 2x2, returns (ct, None).
      * Computes chi-square + expected counts.
      * For 2x2 tables, computes OR using Haldane-Anscombe correction
        when any cell is zero.
      * If 2x2 with min_expected < ``min_expected``, returns Fisher-exact
        result; OR is from ``scipy.stats.fisher_exact``.
      * Otherwise returns chi-square result; OR is the manually computed
        2x2 OR (NaN for >2x2).
    """
    d = df.copy()
    if dropna:
        d = d.dropna(subset=[row, col])

    ct = pd.crosstab(d[row], d[col], dropna=False)

    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ct, None

    chi2, p_chi2, dof, exp = stats.chi2_contingency(ct)
    exp_min = float(np.min(exp))

    odds_ratio: float = np.nan
    if ct.shape == (2, 2):
        ct2 = ct.copy()
        if set(ct2.index).issuperset({0, 1}):
            ct2 = ct2.reindex(index=[0, 1])
        if set(ct2.columns).issuperset({0, 1}):
            ct2 = ct2.reindex(columns=[0, 1])

        a = ct2.iloc[1, 1]
        b = ct2.iloc[1, 0]
        c0 = ct2.iloc[0, 1]
        d0 = ct2.iloc[0, 0]

        # Haldane-Anscombe correction if any zeros
        if min(a, b, c0, d0) == 0:
            a += 0.5
            b += 0.5
            c0 += 0.5
            d0 += 0.5

        odds_ratio = float((a * d0) / (b * c0))

    if ct.shape == (2, 2) and exp_min < min_expected:
        oddsratio_f, p_f = stats.fisher_exact(ct.values)
        return ct, {
            "row": row,
            "col": col,
            "test": "Fisher exact (2x2)",
            "p_value": float(p_f),
            "p_fisher": float(p_f),
            "odds_ratio": float(oddsratio_f),
            "p_chi2": float(p_chi2),
            "chi2": float(chi2),
            "df": int(dof),
            "min_expected": exp_min,
            "note": f"Fisher used (min expected {exp_min:.2f} < {min_expected})",
        }

    return ct, {
        "row": row,
        "col": col,
        "test": "Chi-square",
        "p_value": float(p_chi2),
        "p_chi2": float(p_chi2),
        "chi2": float(chi2),
        "df": int(dof),
        "p_fisher": np.nan,
        "odds_ratio": odds_ratio,
        "min_expected": exp_min,
        "note": "" if exp_min >= min_expected else f"Warning: min expected {exp_min:.2f} < {min_expected}",
    }


def run_chi2_batch(
    df: pd.DataFrame,
    group_col: Optional[str],
    categorical_vars: Sequence[str],
    event_col: str = EVENT_COL,
    min_expected: float = 5,
) -> pd.DataFrame:
    """Run chi2/Fisher for each categorical var vs ``event_col`` within ``df``.

    The ``group_col`` argument is accepted for backward compatibility with
    the original notebook signature and is currently unused by the function
    body (the caller pre-subsets ``df`` for within-group runs).
    """
    rows: List[Dict[str, Any]] = []
    for v in categorical_vars:
        if v not in df.columns:
            continue
        ct, res = chi2_or_fisher_2way(
            df, v, event_col, min_expected=min_expected, dropna=True
        )
        if res is None:
            rows.append(
                {"Variable": v, "test": None, "p_value": np.nan, "note": "not testable"}
            )
        else:
            rows.append(
                {
                    "Variable": v,
                    "test": res["test"],
                    "chi2": res.get("chi2", np.nan),
                    "df": res.get("df", np.nan),
                    "odds_ratio": res.get("odds_ratio", np.nan),
                    "p_value": res.get("p_value", np.nan),
                    "p_chi2": res.get("p_chi2", np.nan),
                    "p_fisher": res.get("p_fisher", np.nan),
                    "min_expected": res.get("min_expected", np.nan),
                    "note": res.get("note", ""),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and "p_value" in out.columns:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out["p_value"] = out["p_value"].round(4)
        out["q_value_fdr_bh"] = out["q_value_fdr_bh"].round(4)
    return out


def run_logrank_binary_batch(
    df: pd.DataFrame,
    binary_vars: Sequence[str],
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """Run a 2-group log-rank test for each binary variable (0 vs 1)."""
    rows: List[Dict[str, Any]] = []
    for v in binary_vars:
        if v not in df.columns:
            continue
        s = pd.to_numeric(df[v], errors="coerce")
        mask0 = s == 0
        mask1 = s == 1
        n0, n1 = int(mask0.sum()), int(mask1.sum())
        if n0 < 2 or n1 < 2:
            rows.append(
                {
                    "Variable": v,
                    "n_0": n0,
                    "n_1": n1,
                    "test_stat": np.nan,
                    "p_logrank": np.nan,
                    "note": "insufficient groups",
                }
            )
            continue
        g0 = df.loc[mask0]
        g1 = df.loc[mask1]
        try:
            res = logrank_test(
                g0[time_col],
                g1[time_col],
                event_observed_A=pd.to_numeric(g0[event_col], errors="coerce").fillna(0),
                event_observed_B=pd.to_numeric(g1[event_col], errors="coerce").fillna(0),
            )
            rows.append(
                {
                    "Variable": v,
                    "n_0": n0,
                    "n_1": n1,
                    "events_0": int(
                        pd.to_numeric(g0[event_col], errors="coerce").fillna(0).sum()
                    ),
                    "events_1": int(
                        pd.to_numeric(g1[event_col], errors="coerce").fillna(0).sum()
                    ),
                    "test_stat": round(float(res.test_statistic), 4),
                    "p_logrank": float(res.p_value),
                    "note": "",
                }
            )
        except Exception as e:  # pragma: no cover - defensive
            rows.append(
                {
                    "Variable": v,
                    "n_0": n0,
                    "n_1": n1,
                    "test_stat": np.nan,
                    "p_logrank": np.nan,
                    "note": str(e),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty and "p_logrank" in out.columns:
        out["q_fdr_bh"] = fdr_bh(out["p_logrank"].values).round(4)
        out["p_logrank"] = out["p_logrank"].round(4)
    return out


def cox_aligned_binary_vars(
    df: pd.DataFrame,
    cox_vars: Optional[Sequence[str]] = None,
    binary_vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select the list of Cox-aligned binary 0/1 covariates present in
    ``df``. Mirrors the candidate-vars logic in cell 29.

    Preference order: ``cox_vars`` > ``binary_vars`` > auto-detect.
    """
    if cox_vars is not None:
        candidate_vars = [v for v in cox_vars if v in df.columns]
    elif binary_vars is not None:
        candidate_vars = [v for v in binary_vars if v in df.columns]
    else:
        candidate_vars = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            u = set(s.dropna().unique().tolist())
            if u.issubset({0, 1}) and len(u) >= 1:
                candidate_vars.append(c)

    out: List[str] = []
    for v in candidate_vars:
        s = pd.to_numeric(df[v], errors="coerce")
        u = set(s.dropna().unique().tolist())
        if u.issubset({0, 1}) and len(u) >= 1:
            out.append(v)
    return out


# =============================================================================
# Kaplan-Meier
# =============================================================================
def km_plot(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    order: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """Kaplan-Meier plot with time converted from days to YEARS.

    Mirrors notebook cell 35 exactly: time = duration_days / 365.25.
    """
    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce") / 365.25  # -> years
    d[event_col] = (
        pd.to_numeric(d[event_col], errors="coerce").fillna(0).astype(int)
    )
    d = d.dropna(subset=[time_col, group_col])
    d = d[d[time_col] >= 0]
    if d.empty:
        print("No data to plot.")
        return

    kmf = KaplanMeierFitter()
    plt.figure(figsize=figsize)

    groups = list(order) if order is not None else list(pd.Series(d[group_col].unique()))
    for grp in groups:
        g = d[d[group_col] == grp]
        if g.empty:
            continue
        kmf.fit(g[time_col], event_observed=g[event_col], label=f"{grp} (n={len(g)})")
        kmf.plot(ci_show=False)

    plt.title(title)
    plt.xlabel("Years")
    plt.ylabel("Survival probability")
    plt.grid(True, alpha=0.25)
    plt.show()


# =============================================================================
# Multi-group + pairwise log-rank (reviewer #81)
# =============================================================================
def multi_group_logrank_summary(
    df: pd.DataFrame,
    group_col: str,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> Tuple[pd.DataFrame, Any]:
    """Per-group counts/survival% and the multivariate log-rank result.

    Returns
    -------
    grp_summary : DataFrame with ``Group``, ``Episodes (n)``, ``Failures (n)``,
        ``Survival (%)``.
    lr_all : the lifelines ``StatisticalResult`` from
        :func:`multivariate_logrank_test`.
    """
    rows = []
    for grp, sub in df.groupby(group_col):
        n = len(sub)
        fail = int(sub[event_col].sum())
        rows.append(
            {
                "Group": grp,
                "Episodes (n)": n,
                "Failures (n)": fail,
                "Survival (%)": f"{100*(1 - fail/n):.2f}",
            }
        )
    grp_summary = pd.DataFrame(rows)
    lr_all = multivariate_logrank_test(df[time_col], df[group_col], df[event_col])
    return grp_summary, lr_all


def pairwise_logrank_with_bh(
    df: pd.DataFrame,
    group_col: str,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """Pairwise 2-sample log-rank tests with BH-corrected q-values.
    Mirrors the pairwise loop in cell 37."""
    groups = list(df[group_col].dropna().unique())
    rows = []
    for g1, g2 in combinations(groups, 2):
        s1 = df[df[group_col] == g1]
        s2 = df[df[group_col] == g2]
        res = logrank_test(
            s1[time_col],
            s2[time_col],
            event_observed_A=s1[event_col],
            event_observed_B=s2[event_col],
        )
        rows.append(
            {"Group A": g1, "Group B": g2, "p (log-rank)": round(res.p_value, 6)}
        )
    pairwise_df = pd.DataFrame(rows)
    if not pairwise_df.empty:
        pairwise_df["q (BH)"] = fdr_bh(
            pairwise_df["p (log-rank)"].tolist()
        ).round(6)
    return pairwise_df


# =============================================================================
# Cox proportional hazards
# =============================================================================
def _cox_hr_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Clean Cox output table with HR and 95% CI, sorted by p-value."""
    s = cph.summary.copy()
    required = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise KeyError(
            f"Cox summary missing columns: {missing}. Available: {list(s.columns)}"
        )

    out = pd.DataFrame(
        {
            "Variable": s.index.astype(str),
            "HR": s["exp(coef)"].astype(float),
            "95% CI (lower)": s["exp(coef) lower 95%"].astype(float),
            "95% CI (upper)": s["exp(coef) upper 95%"].astype(float),
            "p_cox": s["p"].astype(float),
        }
    )
    out["HR"] = out["HR"].round(3)
    out["95% CI (lower)"] = out["95% CI (lower)"].round(3)
    out["95% CI (upper)"] = out["95% CI (upper)"].round(3)
    out["p_cox"] = out["p_cox"].round(4)
    return out.sort_values("p_cox")


def build_cox_matrix(
    df: pd.DataFrame,
    include_cols: Sequence[str],
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    min_positive: int = 20,
    categorical_cols: Sequence[str] = ("Cohort", "Coronal_Restoration_Group", "AgeGroup"),
    ref_cols: Sequence[str] = (
        "Cohort_Root canal treatment",
        "Coronal_Restoration_Group_Neither",
        "AgeGroup_40–60",
        "AgeGroup_40-60",
    ),
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Build the Cox design matrix.

    Mirrors notebook cell 39 ``build_cox_matrix`` exactly, including:
      * Coerce time/event to numeric, drop NA, keep duration >= 0.
      * Skip columns with <=1 unique value or 0/1 columns with fewer than
        ``min_positive`` positives.
      * One-hot encode the categorical columns with ``drop_first=False``,
        then drop the reference dummies in ``ref_cols``.
      * Drop any constant columns that survive.

    Returns ``(X, error_message)`` where exactly one is None.
    """
    d = df.copy()
    d[time_col] = pd.to_numeric(d.get(time_col), errors="coerce")
    d[event_col] = (
        pd.to_numeric(d.get(event_col), errors="coerce").fillna(0).astype(int)
    )
    d = d.dropna(subset=[time_col, event_col])
    d = d[d[time_col] >= 0]

    if d[event_col].sum() == 0:
        return None, "No events -> Cox model not estimable."

    keep: List[str] = []
    for c in include_cols:
        if c not in d.columns:
            continue
        if str(d[c].dtype) == "category" or d[c].dtype == "object":
            keep.append(c)
            continue
        if d[c].nunique(dropna=True) <= 1:
            continue
        if (
            set(pd.to_numeric(d[c], errors="coerce").dropna().unique()).issubset({0, 1})
            and (pd.to_numeric(d[c], errors="coerce") == 1).sum() < min_positive
        ):
            continue
        keep.append(c)

    X = d[[time_col, event_col] + keep].copy()

    for cat in categorical_cols:
        if cat in X.columns:
            X[cat] = X[cat].astype("category")

    X = pd.get_dummies(
        X,
        columns=[c for c in categorical_cols if c in X.columns],
        drop_first=False,
    )
    X = X.drop(columns=[c for c in ref_cols if c in X.columns], errors="ignore")

    const_cols = [
        c
        for c in X.columns
        if c not in [time_col, event_col] and X[c].nunique(dropna=True) <= 1
    ]
    X = X.drop(columns=const_cols, errors="ignore")

    return X, None


def fit_cox(
    df: pd.DataFrame,
    include_cols: Sequence[str],
    penalizer: float = 0.1,
    min_positive: int = 20,
    cluster_col: str = ID_COL,
    robust: bool = True,
    check_ph: bool = True,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> Tuple[Optional[CoxPHFitter], Optional[pd.DataFrame], Optional[str]]:
    """Fit a Cox PH model with optional cluster-robust SE and PH check.

    Mirrors notebook cell 39 ``fit_cox`` exactly. Returns
    ``(cph, hr_table, error_message)``.
    """
    X, err = build_cox_matrix(
        df,
        include_cols,
        time_col=time_col,
        event_col=event_col,
        min_positive=min_positive,
    )
    if err:
        return None, None, err
    if X is None or X.empty:
        return None, None, "Empty design matrix."

    # lifelines can fail on pandas extension/object dtypes (e.g., nullable Int64).
    # Normalize to plain numeric dtypes before fitting.
    X_fit = X.copy()
    X_fit[time_col] = pd.to_numeric(X_fit[time_col], errors="coerce")
    X_fit[event_col] = pd.to_numeric(X_fit[event_col], errors="coerce").fillna(0).astype(int)
    covariate_cols = [c for c in X_fit.columns if c not in [time_col, event_col]]
    for c in covariate_cols:
        X_fit[c] = pd.to_numeric(X_fit[c], errors="coerce")

    X_fit = X_fit.dropna(subset=[time_col, event_col] + covariate_cols)
    if X_fit.empty:
        return None, None, "No complete rows remain after Cox dtype sanitization."

    cph = CoxPHFitter(penalizer=penalizer)

    fit_kwargs = dict(duration_col=time_col, event_col=event_col)
    if cluster_col in df.columns:
        # Align cluster series to X.index (X derived from df with drops)
        cluster_series = df.loc[X_fit.index, cluster_col]
        cluster_series = cluster_series.astype("string").fillna("<missing>").astype(str)
        X_fit[cluster_col] = cluster_series.values
        cph.fit(X_fit, **fit_kwargs, robust=robust, cluster_col=cluster_col)
    else:
        cph.fit(X_fit, **fit_kwargs)

    tbl = _cox_hr_table(cph)

    # PH check (diagnostics)
    if check_ph:
        try:
            print("PH assumption check (lifelines):")
            ph_cols = [c for c in X.columns if c != cluster_col]
            cph.check_assumptions(
                X_fit if cluster_col in df.columns else X,
                p_value_threshold=0.05,
                show_plots=False,
                columns=ph_cols,
            )
        except Exception as e:  # pragma: no cover
            print("PH check failed (non-fatal):", e)

    return cph, tbl, None


# =============================================================================
# Forest plot (reviewer #84)
# =============================================================================
def plot_cox_forest(
    hr_table: pd.DataFrame,
    title: str = (
        "Adjusted Hazard Ratios for Tooth Extraction Following Primary Root Canal Treatment\n"
        "Multivariable Cox Proportional Hazards Model (n = 119,762 Episodes)"
    ),
    label_map: Optional[Dict[str, str]] = None,
    reference_note: str = "Reference categories: No coronal restoration; Age 40–60 years; Female",
    output_path: Optional[str] = "forest_plot_primary_rct.png",
    figsize: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Publication-ready forest plot from a Cox HR table.

    Mirrors notebook cell 42 with the same defaults (manuscript title,
    label map, reference note, save path). Returns the sorted dataframe
    used to draw the figure.
    """
    if label_map is None:
        label_map = FOREST_LABEL_MAP

    df_fp = hr_table.copy().reset_index(drop=True)
    df_fp = df_fp.sort_values("HR", ascending=True).reset_index(drop=True)
    df_fp["Variable_clean"] = df_fp["Variable"].map(label_map).fillna(df_fp["Variable"])

    if figsize is None:
        figsize = (9, max(5, len(df_fp) * 0.48))
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in df_fp.iterrows():
        color = "steelblue" if row["HR"] <= 1 else "firebrick"
        ax.hlines(
            i,
            row["95% CI (lower)"],
            row["95% CI (upper)"],
            color=color,
            linewidth=1.8,
            zorder=2,
        )
        ax.plot(
            row["HR"],
            i,
            "o",
            color=color,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.9, zorder=1)

    ax.set_yticks(range(len(df_fp)))
    ax.set_yticklabels(df_fp["Variable_clean"].tolist(), fontsize=9)

    x_max = df_fp["95% CI (upper)"].max()
    for i, row in df_fp.iterrows():
        sig = "**" if row["p_cox"] < 0.001 else ("*" if row["p_cox"] < 0.05 else "")
        ax.text(
            x_max * 1.02,
            i,
            f"{row['HR']:.3f} ({row['95% CI (lower)']:.3f}\u2013{row['95% CI (upper)']:.3f}){sig}",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.set_xlabel("Hazard Ratio (HR) with 95% Confidence Intervals", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(
        left=max(0.5, df_fp["95% CI (lower)"].min() * 0.88),
        right=x_max * 1.35,
    )

    if reference_note:
        fig.text(0.5, -0.02, reference_note, ha="center", fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return df_fp


# =============================================================================
# PROBE / reviewer audit (cell 19)
# =============================================================================
# All `_*` helpers below were extracted verbatim from cell 19. The
# orchestrator `compute_probe_tables` returns a dict of DataFrames and
# pre-computed counts; the notebook handles `display(...)` so output
# order is preserved without notebook-only constructs in this module.

def _has_cols(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(col in df.columns for col in cols)


def _safe_nunique(df: pd.DataFrame, col: str) -> Any:
    return int(df[col].dropna().nunique()) if col in df.columns else np.nan


def _safe_sum(series_like: Any) -> int:
    s = pd.to_numeric(series_like, errors="coerce")
    return int(s.fillna(0).sum()) if len(s) else 0


def _safe_missing(series_like: Any) -> int:
    return int(pd.isna(series_like).sum())


def _fmt_pct(n: Any, denom: Any) -> Any:
    if denom in [None, 0] or pd.isna(denom):
        return np.nan
    return round(float(100.0 * n / denom), 2)


def _series_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="object")


def _ordered_unique(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _derive_rct_from_episodes(episodes_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(episodes_df, pd.DataFrame) or "Cohort" not in episodes_df.columns:
        return pd.DataFrame()
    cohort_labels = episodes_df["Cohort"].astype(str).str.strip()
    return episodes_df.loc[cohort_labels == "Root canal treatment"].copy()


def _rct_key_set(df: pd.DataFrame) -> Optional[set]:
    candidate_keys = [
        ["episode_id"],
        ["Patient_ID", "Tooth_Num", "Cohort", "start_date"],
        ["Patient_ID", "Tooth_Num", "Cohort"],
    ]
    for key_cols in candidate_keys:
        if _has_cols(df, key_cols):
            return set(
                map(
                    tuple,
                    df[key_cols]
                    .astype(str)
                    .fillna("<NA>")
                    .drop_duplicates()
                    .itertuples(index=False, name=None),
                )
            )
    return None


def _resolve_rct(
    episodes_df: pd.DataFrame, existing_rct: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, str]:
    """Return (rct_df, source_note). If ``existing_rct`` is provided and
    matches the derived RCT subset, prefer it; else rederive from
    ``episodes_df``."""
    derived = _derive_rct_from_episodes(episodes_df)

    if not isinstance(existing_rct, pd.DataFrame) or existing_rct.empty:
        return derived, "rederived from episodes"

    if "Cohort" not in existing_rct.columns:
        return derived, "rederived from episodes (existing rct missing Cohort)"

    existing_candidate = existing_rct.copy()
    existing_candidate["Cohort"] = existing_candidate["Cohort"].astype(str).str.strip()
    if (existing_candidate["Cohort"] != "Root canal treatment").any():
        return derived, "rederived from episodes (existing rct contains non-RCT rows)"

    if derived.empty:
        return existing_candidate, "used existing rct (episodes subset unavailable)"

    existing_keys = _rct_key_set(existing_candidate)
    derived_keys = _rct_key_set(derived)
    if (
        existing_keys is not None
        and derived_keys is not None
        and existing_keys == derived_keys
    ):
        return existing_candidate, "used validated existing rct"

    if len(existing_candidate) == len(derived):
        return existing_candidate, "used existing rct (matched derived row count)"

    return derived, "rederived from episodes (existing rct did not match derived subset)"


def _resolve_model_vars(
    rct_df: pd.DataFrame,
    covars_rct: Optional[Sequence[str]] = None,
    base_covars: Optional[Sequence[str]] = None,
    requested_default: Optional[Sequence[str]] = None,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> Tuple[str, List[str], List[str]]:
    """Pick the best available model variable list for the complete-case
    audit. Mirrors cell-19 logic."""
    if requested_default is None:
        requested_default = REQUESTED_DEFAULT_MODEL_VARS

    candidate_specs: List[Tuple[str, List[str]]] = []
    if isinstance(covars_rct, list):
        candidate_specs.append(
            ("covars_rct + survival vars", [time_col, event_col] + list(covars_rct))
        )
    if isinstance(base_covars, list):
        candidate_specs.append(
            (
                "base_covars + survival vars",
                [time_col, event_col]
                + [col for col in base_covars if col != "Cohort"],
            )
        )
    candidate_specs.append(("requested default variable list", list(requested_default)))

    for basis, candidate_vars in candidate_specs:
        ordered = _ordered_unique(candidate_vars)
        available = [col for col in ordered if col in rct_df.columns]
        if available:
            return basis, ordered, available

    return "no model variables available", list(requested_default), []


def _fallback_assign_cohort(text: Any) -> Optional[str]:
    """Identical to :func:`assign_cohort`; kept under the cell-19 name
    for parity if a downstream caller passes it explicitly."""
    return assign_cohort(text)


def _prepare_raw_for_probe(
    raw_df: pd.DataFrame,
    assign_cohort_fn: Optional[Callable[[Any], Optional[str]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure raw has ``Treatment_Date_dt``/``Failure_Treatment_Date_dt``/
    ``Cohort`` for the audit. Returns (raw_local, prep_notes)."""
    if not isinstance(raw_df, pd.DataFrame):
        return pd.DataFrame(), []

    raw_local = raw_df.copy()
    prep_notes: List[str] = []

    if (
        "Treatment_Date_dt" not in raw_local.columns
        and "Treatment_Date" in raw_local.columns
    ):
        raw_local["Treatment_Date_dt"] = pd.to_datetime(
            raw_local["Treatment_Date"], errors="coerce", dayfirst=True
        )
        prep_notes.append("derived Treatment_Date_dt from Treatment_Date")

    if (
        "Failure_Treatment_Date_dt" not in raw_local.columns
        and "Failure_Treatment_Date" in raw_local.columns
    ):
        raw_local["Failure_Treatment_Date_dt"] = pd.to_datetime(
            raw_local["Failure_Treatment_Date"], errors="coerce", dayfirst=True
        )
        prep_notes.append("derived Failure_Treatment_Date_dt from Failure_Treatment_Date")

    if assign_cohort_fn is None or not callable(assign_cohort_fn):
        assign_cohort_fn = _fallback_assign_cohort

    if "סיווג" in raw_local.columns:
        derived_cohort = (
            raw_local["סיווג"].fillna("").astype(str).apply(assign_cohort_fn)
        )
        if "Cohort" not in raw_local.columns:
            raw_local["Cohort"] = derived_cohort
            prep_notes.append("derived Cohort from סיווג")
        elif raw_local["Cohort"].isna().any():
            raw_local["Cohort"] = raw_local["Cohort"].where(
                raw_local["Cohort"].notna(), derived_cohort
            )
            prep_notes.append("filled missing Cohort values from סיווג")

    return raw_local, prep_notes


def _rebuild_prefilter_episode_frame(
    raw_df: pd.DataFrame, study_end_value: pd.Timestamp
) -> pd.DataFrame:
    """Rebuild an episode-level frame from raw using the same logic as
    :func:`build_episodes` but BEFORE the duration drops, so we can
    diagnose pre-filter issues. Mirrors cell-19 helper."""
    required = ["Patient_ID", "Tooth_Num", "Treatment_Date_dt", "Cohort"]
    if not isinstance(raw_df, pd.DataFrame) or not _has_cols(raw_df, required):
        return pd.DataFrame()

    pre_rows = raw_df.dropna(subset=required).copy()
    if pre_rows.empty:
        return pd.DataFrame()

    starts_local = (
        pre_rows.groupby(["Patient_ID", "Tooth_Num", "Cohort"], as_index=False)
        .agg(start_date=("Treatment_Date_dt", "min"))
    )

    if "Failure_Treatment_Date_dt" in pre_rows.columns:
        fail_tmp = pre_rows.merge(
            starts_local,
            on=["Patient_ID", "Tooth_Num", "Cohort"],
            how="inner",
        )
        fail_tmp = fail_tmp[fail_tmp["Failure_Treatment_Date_dt"].notna()].copy()
        fail_tmp = fail_tmp[
            fail_tmp["Failure_Treatment_Date_dt"] >= fail_tmp["start_date"]
        ].copy()
        min_fail_local = (
            fail_tmp.groupby(["Patient_ID", "Tooth_Num", "Cohort"], as_index=False)
            .agg(failure_date=("Failure_Treatment_Date_dt", "min"))
        )
    else:
        min_fail_local = starts_local[["Patient_ID", "Tooth_Num", "Cohort"]].copy()
        min_fail_local["failure_date"] = pd.NaT

    prefilter = starts_local.merge(
        min_fail_local,
        on=["Patient_ID", "Tooth_Num", "Cohort"],
        how="left",
    )
    prefilter["event"] = prefilter["failure_date"].notna().astype(int)
    prefilter["stop_date"] = pd.to_datetime(
        prefilter["failure_date"].fillna(study_end_value), errors="coerce"
    )
    prefilter["duration_days"] = (
        pd.to_datetime(prefilter["stop_date"], errors="coerce")
        - pd.to_datetime(prefilter["start_date"], errors="coerce")
    ).dt.days
    return prefilter


def compute_probe_tables(
    raw: pd.DataFrame,
    episodes: pd.DataFrame,
    study_end: pd.Timestamp = STUDY_END,
    rct: Optional[pd.DataFrame] = None,
    covars_rct: Optional[Sequence[str]] = None,
    base_covars: Optional[Sequence[str]] = None,
    expected_counts: Optional[Dict[str, int]] = None,
    expected_resto_groups: Optional[Sequence[str]] = None,
    requested_default_model_vars: Optional[Sequence[str]] = None,
    excel_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the PROBE / reviewer audit tables and manuscript text.

    Mirrors notebook cell 19 exactly in the values it computes. Returns
    a dict with all tables, key counts, manuscript text strings, and a
    list of warnings to print. The notebook is responsible for
    ``display()`` calls so the output order matches the original.

    Parameters
    ----------
    raw, episodes : the cleaned source rows and final analytic episode
        DataFrame.
    study_end : censoring date (defaults to :data:`STUDY_END`).
    rct : optional pre-built RCT-only subset to validate against the
        derived one.
    covars_rct, base_covars : optional candidate covariate lists for the
        complete-case audit.
    excel_output_path : if not None, also write a multi-sheet workbook.

    Returns
    -------
    dict with keys: ``cohort_flow``, ``diagnostic_exclusions``,
    ``missing_data``, ``restoration_counts``, ``followup_checks``,
    ``model_complete_case_table``, ``model_missing_by_var``, plus
    pre-formatted text and warnings, and a ``rct`` slice.
    """
    if expected_counts is None:
        expected_counts = EXPECTED_COUNTS
    if expected_resto_groups is None:
        expected_resto_groups = EXPECTED_RESTO_GROUPS
    if requested_default_model_vars is None:
        requested_default_model_vars = REQUESTED_DEFAULT_MODEL_VARS

    if not isinstance(episodes, pd.DataFrame):
        raise RuntimeError(
            "compute_probe_tables requires a finalized `episodes` DataFrame."
        )
    if not isinstance(raw, pd.DataFrame):
        raise RuntimeError("compute_probe_tables requires the `raw` DataFrame.")

    raw_for_probe, raw_prep_notes = _prepare_raw_for_probe(raw, assign_cohort)
    rct_resolved, rct_source_note = _resolve_rct(episodes, existing_rct=rct)
    if rct_resolved.empty:
        raise RuntimeError("Could not derive the primary RCT cohort from `episodes`.")

    if "event" in rct_resolved.columns:
        rct_resolved["event"] = pd.to_numeric(rct_resolved["event"], errors="coerce")
    if "duration_days" in rct_resolved.columns:
        rct_resolved["duration_days"] = pd.to_numeric(
            rct_resolved["duration_days"], errors="coerce"
        )

    rct_episode_n = int(len(rct_resolved))
    rct_patient_n = _safe_nunique(rct_resolved, "Patient_ID")
    if _has_cols(rct_resolved, ["Patient_ID", "Tooth_Num"]):
        rct_unique_patient_tooth_n = int(
            rct_resolved[["Patient_ID", "Tooth_Num"]].drop_duplicates().shape[0]
        )
    else:
        rct_unique_patient_tooth_n = np.nan
    rct_failure_n = _safe_sum(_series_or_na(rct_resolved, "event") == 1)
    rct_censored_n = _safe_sum(_series_or_na(rct_resolved, "event") == 0)

    warnings: List[str] = []
    if rct_episode_n != expected_counts["episodes"]:
        warnings.append(
            f"WARNING: primary RCT episodes differ from manuscript expectation "
            f"({rct_episode_n:,} vs {expected_counts['episodes']:,})."
        )
    if (not pd.isna(rct_patient_n)) and rct_patient_n != expected_counts["patients"]:
        warnings.append(
            f"WARNING: primary RCT patients differ from manuscript expectation "
            f"({rct_patient_n:,} vs {expected_counts['patients']:,})."
        )
    if rct_failure_n != expected_counts["failures"]:
        warnings.append(
            f"WARNING: primary RCT failures differ from manuscript expectation "
            f"({rct_failure_n:,} vs {expected_counts['failures']:,})."
        )
    if rct_censored_n != expected_counts["censored"]:
        warnings.append(
            f"WARNING: primary RCT censored/survived differ from manuscript expectation "
            f"({rct_censored_n:,} vs {expected_counts['censored']:,})."
        )

    prefilter_episodes = _rebuild_prefilter_episode_frame(raw_for_probe, study_end)

    eligible_required = ["Patient_ID", "Tooth_Num", "Treatment_Date_dt", "Cohort"]
    raw_rows_n = int(len(raw_for_probe))
    raw_eligible_rows_n = (
        int(
            raw_for_probe.dropna(
                subset=[c for c in eligible_required if c in raw_for_probe.columns]
            ).shape[0]
        )
        if _has_cols(raw_for_probe, eligible_required)
        else np.nan
    )
    if _has_cols(raw_for_probe, eligible_required):
        eligible_rows = raw_for_probe.dropna(subset=eligible_required).copy()
        pre_episode_combo_n = int(
            eligible_rows[["Patient_ID", "Tooth_Num", "Cohort"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        pre_episode_combo_n = np.nan

    cohort_flow = pd.DataFrame(
        [
            {
                "Stage": "Raw rows in source file",
                "n": raw_rows_n,
                "Notes": "Row-level records read from the source Excel file.",
            },
            {
                "Stage": "Rows eligible for episode construction",
                "n": raw_eligible_rows_n,
                "Notes": "Rows with non-missing Patient_ID, Tooth_Num, Treatment_Date_dt, and Cohort.",
            },
            {
                "Stage": "Unique Patient_ID × Tooth_Num × Cohort combinations before follow-up filtering",
                "n": pre_episode_combo_n,
                "Notes": "Episode keys before duration-based filtering.",
            },
            {
                "Stage": "Episode-level records after duration/follow-up filtering",
                "n": int(len(episodes)),
                "Notes": "Final analytic episode dataset after dropping missing or negative duration_days.",
            },
            {
                "Stage": "Primary RCT episodes",
                "n": rct_episode_n,
                "Notes": "Episodes with Cohort = Root canal treatment.",
            },
            {
                "Stage": "Primary RCT failures/extractions",
                "n": rct_failure_n,
                "Notes": "Primary RCT episodes with event = 1.",
            },
            {
                "Stage": "Primary RCT censored/survived",
                "n": rct_censored_n,
                "Notes": "Primary RCT episodes with event = 0.",
            },
        ]
    )

    model_basis, model_vars_requested, model_vars_available = _resolve_model_vars(
        rct_resolved,
        covars_rct=covars_rct,
        base_covars=base_covars,
        requested_default=requested_default_model_vars,
    )
    model_missing_any = (
        int(rct_resolved[model_vars_available].isna().any(axis=1).sum())
        if model_vars_available
        else np.nan
    )

    diagnostic_rows: List[Dict[str, Any]] = []
    raw_denominator = int(len(raw_for_probe))
    for col in ["Patient_ID", "Tooth_Num", "Treatment_Date_dt", "Cohort"]:
        n_missing = _safe_missing(_series_or_na(raw_for_probe, col))
        diagnostic_rows.append(
            {
                "Check": f"Raw rows missing {col}",
                "n": n_missing,
                "Denominator": raw_denominator,
                "Interpretation": f"Source-row diagnostic before cohort construction for {col}, using current notebook-consistent audit fields.",
                "True exclusion yes/no/unknown": "unknown",
            }
        )

    prefilter_denominator = int(len(prefilter_episodes))
    prefilter_duration_missing_n = (
        _safe_missing(_series_or_na(prefilter_episodes, "duration_days"))
        if prefilter_denominator
        else np.nan
    )
    prefilter_duration_negative_n = (
        int(
            (
                pd.to_numeric(
                    _series_or_na(prefilter_episodes, "duration_days"), errors="coerce"
                )
                < 0
            )
            .fillna(False)
            .sum()
        )
        if prefilter_denominator
        else np.nan
    )
    diagnostic_rows.append(
        {
            "Check": "Episode-level records missing duration_days before duration filtering",
            "n": prefilter_duration_missing_n,
            "Denominator": prefilter_denominator,
            "Interpretation": "Prefilter episode diagnostic reconstructed from the current raw-to-episode logic.",
            "True exclusion yes/no/unknown": "yes",
        }
    )
    diagnostic_rows.append(
        {
            "Check": "Episode-level records with duration_days < 0 before duration filtering",
            "n": prefilter_duration_negative_n,
            "Denominator": prefilter_denominator,
            "Interpretation": "Prefilter episode diagnostic reconstructed from the current raw-to-episode logic.",
            "True exclusion yes/no/unknown": "yes",
        }
    )

    rct_denominator = int(len(rct_resolved))
    rct_duration_missing_n = _safe_missing(_series_or_na(rct_resolved, "duration_days"))
    rct_duration_negative_n = int(
        (
            pd.to_numeric(_series_or_na(rct_resolved, "duration_days"), errors="coerce")
            < 0
        )
        .fillna(False)
        .sum()
    )
    failure_mask = pd.to_numeric(_series_or_na(rct_resolved, "event"), errors="coerce") == 1
    censor_mask = pd.to_numeric(_series_or_na(rct_resolved, "event"), errors="coerce") == 0
    failure_denominator = int(failure_mask.sum())
    censor_denominator = int(censor_mask.sum())
    failure_date_missing_in_failures_n = (
        _safe_missing(_series_or_na(rct_resolved.loc[failure_mask], "failure_date"))
        if failure_denominator
        else np.nan
    )
    stop_date_missing_in_censored_n = (
        _safe_missing(_series_or_na(rct_resolved.loc[censor_mask], "stop_date"))
        if censor_denominator
        else np.nan
    )

    if "Coronal_Restoration_Group" in rct_resolved.columns:
        resto_missing_or_unclassified_n = int(
            (
                rct_resolved["Coronal_Restoration_Group"].isna()
                | ~rct_resolved["Coronal_Restoration_Group"].isin(
                    expected_resto_groups
                )
            ).sum()
        )
    else:
        resto_missing_or_unclassified_n = np.nan

    diagnostic_rows.extend(
        [
            {
                "Check": "Primary RCT episodes missing duration_days",
                "n": rct_duration_missing_n,
                "Denominator": rct_denominator,
                "Interpretation": "Diagnostic within the final analytical cohort.",
                "True exclusion yes/no/unknown": "no",
            },
            {
                "Check": "Primary RCT episodes with duration_days < 0",
                "n": rct_duration_negative_n,
                "Denominator": rct_denominator,
                "Interpretation": "Diagnostic within the final analytical cohort; should be zero after filtering.",
                "True exclusion yes/no/unknown": "no",
            },
            {
                "Check": "Failures/extractions missing failure_date",
                "n": failure_date_missing_in_failures_n,
                "Denominator": failure_denominator,
                "Interpretation": "Assessed only among event = 1 episodes; failure_date is structurally required here.",
                "True exclusion yes/no/unknown": "unknown",
            },
            {
                "Check": "Censored episodes missing stop_date",
                "n": stop_date_missing_in_censored_n,
                "Denominator": censor_denominator,
                "Interpretation": "Assessed only among event = 0 episodes; stop_date should reflect censoring date.",
                "True exclusion yes/no/unknown": "unknown",
            },
            {
                "Check": "Missing or unclassifiable Coronal_Restoration_Group",
                "n": resto_missing_or_unclassified_n,
                "Denominator": rct_denominator,
                "Interpretation": "Diagnostic classification check for the restoration grouping variable.",
                "True exclusion yes/no/unknown": "unknown",
            },
            {
                "Check": "Primary RCT episodes missing at least one model covariate",
                "n": model_missing_any,
                "Denominator": rct_denominator,
                "Interpretation": f"True exclusion from a complete-case multivariable model using {model_basis}.",
                "True exclusion yes/no/unknown": "yes",
            },
        ]
    )

    diagnostic_exclusions = pd.DataFrame(diagnostic_rows)

    missing_specs = [
        ("Core/survival", "Patient_ID", None, "Analytical cohort identifier."),
        ("Core/survival", "Tooth_Num", None, "Analytical cohort tooth identifier."),
        ("Core/survival", "Cohort", None, "Cohort should be fixed to primary root canal treatment in this subset."),
        ("Core/survival", "start_date", None, "Episode start date."),
        ("Core/survival", "stop_date", None, "Required for both failures and censored episodes in the analytical cohort."),
        ("Core/survival", "duration_days", None, "Primary follow-up time variable."),
        ("Core/survival", "event", None, "Binary event indicator."),
        ("Core/survival", "failure_date", "event == 1", "Assessed only among failures/extractions; not required for censored teeth."),
        ("Patient/descriptive/model", "Age", None, "Age at treatment, if available."),
        ("Patient/descriptive/model", "AgeGroup", None, "Derived age group used in descriptive/model tables, if available."),
        ("Patient/descriptive/model", "Male", None, "Sex indicator, if available."),
        ("Patient/descriptive/model", "Smoking", None, "Patient-level covariate, if available."),
        ("Patient/descriptive/model", "Cancer", None, "Patient-level covariate, if available."),
        ("Patient/descriptive/model", "Diabetes", None, "Patient-level covariate, if available."),
        ("Patient/descriptive/model", "Hypertension", None, "Patient-level covariate, if available."),
        ("Patient/descriptive/model", "Biphos_use", None, "Patient-level covariate, if available."),
        ("Restoration", "Has_Sealing_in_window", None, "Absence of sealing is coded as 0 and is not missing."),
        ("Restoration", "Has_Crown_in_window", None, "Absence of crown is coded as 0 and is not missing."),
        ("Restoration", "Coronal_Restoration_Group", None, "Expected groups are Neither, Sealing only, and Sealing + Crown."),
    ]

    missing_rows: List[Dict[str, Any]] = []
    for section, variable, condition, note in missing_specs:
        if variable not in rct_resolved.columns:
            missing_rows.append(
                {
                    "Section": section,
                    "Variable": variable,
                    "Denominator": np.nan,
                    "Missing n": np.nan,
                    "Missing %": np.nan,
                    "Notes": f"Column not present in analytical cohort. {note}",
                }
            )
            continue
        if condition == "event == 1":
            sub = rct_resolved.loc[failure_mask].copy()
            denominator = int(len(sub))
        else:
            sub = rct_resolved
            denominator = int(len(sub))
        missing_n = _safe_missing(sub[variable])
        missing_rows.append(
            {
                "Section": section,
                "Variable": variable,
                "Denominator": denominator,
                "Missing n": missing_n,
                "Missing %": _fmt_pct(missing_n, denominator),
                "Notes": note,
            }
        )

    missing_data = pd.DataFrame(missing_rows)

    restoration_rows: List[Dict[str, Any]] = []
    if "Coronal_Restoration_Group" in rct_resolved.columns:
        resto_counts = rct_resolved["Coronal_Restoration_Group"].value_counts(
            dropna=False
        )
        for group_name in expected_resto_groups:
            count = int(resto_counts.get(group_name, 0))
            restoration_rows.append(
                {
                    "Coronal_Restoration_Group": group_name,
                    "n": count,
                    "%": _fmt_pct(count, rct_denominator),
                }
            )
        missing_resto_n = int(rct_resolved["Coronal_Restoration_Group"].isna().sum())
        unclassified_resto_n = int(
            (
                rct_resolved["Coronal_Restoration_Group"].notna()
                & ~rct_resolved["Coronal_Restoration_Group"].isin(expected_resto_groups)
            ).sum()
        )
        restoration_rows.append(
            {
                "Coronal_Restoration_Group": "Missing",
                "n": missing_resto_n,
                "%": _fmt_pct(missing_resto_n, rct_denominator),
            }
        )
        restoration_rows.append(
            {
                "Coronal_Restoration_Group": "Unclassifiable",
                "n": unclassified_resto_n,
                "%": _fmt_pct(unclassified_resto_n, rct_denominator),
            }
        )
    else:
        restoration_rows.append(
            {"Coronal_Restoration_Group": "Column not present", "n": np.nan, "%": np.nan}
        )

    restoration_counts = pd.DataFrame(restoration_rows)

    duration_series = pd.to_numeric(
        _series_or_na(rct_resolved, "duration_days"), errors="coerce"
    )
    followup_checks = pd.DataFrame(
        [
            {"Metric": "Mean duration_days", "Value": duration_series.mean()},
            {"Metric": "SD duration_days", "Value": duration_series.std()},
            {"Metric": "Median duration_days", "Value": duration_series.median()},
            {
                "Metric": "IQR duration_days",
                "Value": duration_series.quantile(0.75) - duration_series.quantile(0.25),
            },
            {"Metric": "Min duration_days", "Value": duration_series.min()},
            {"Metric": "Max duration_days", "Value": duration_series.max()},
            {"Metric": "Missing duration_days", "Value": int(duration_series.isna().sum())},
            {
                "Metric": "Zero duration_days",
                "Value": int((duration_series == 0).fillna(False).sum()),
            },
            {
                "Metric": "Negative duration_days",
                "Value": int((duration_series < 0).fillna(False).sum()),
            },
            {
                "Metric": "Failures missing failure_date",
                "Value": failure_date_missing_in_failures_n,
            },
            {
                "Metric": "Censored episodes missing stop_date",
                "Value": stop_date_missing_in_censored_n,
            },
        ]
    )

    complete_case_mask = (
        ~rct_resolved[model_vars_available].isna().any(axis=1)
        if model_vars_available
        else pd.Series([True] * len(rct_resolved), index=rct_resolved.index)
    )
    model_complete_n = int(complete_case_mask.sum())
    model_excluded_n = int((~complete_case_mask).sum())

    model_complete_case_table = pd.DataFrame(
        [
            {"Metric": "Primary RCT episodes", "n": rct_denominator},
            {"Metric": "Episodes complete for all model variables", "n": model_complete_n},
            {
                "Metric": "Episodes excluded from model due to missing model variables",
                "n": model_excluded_n,
            },
        ]
    )

    model_missing_by_var_rows: List[Dict[str, Any]] = []
    for variable in model_vars_requested:
        if variable in rct_resolved.columns:
            missing_n = _safe_missing(rct_resolved[variable])
            denominator = rct_denominator
            note = (
                f"Included in the complete-case audit ({model_basis})."
                if variable in model_vars_available
                else f"Available in cohort but not selected for the complete-case audit ({model_basis})."
            )
        else:
            missing_n = np.nan
            denominator = np.nan
            note = (
                f"Column not present in analytical cohort; not counted as missing complete-case data ({model_basis})."
            )
        model_missing_by_var_rows.append(
            {
                "Variable": variable,
                "Missing n": missing_n,
                "Missing %": _fmt_pct(missing_n, denominator),
                "Denominator": denominator,
                "Notes": note,
            }
        )
    model_missing_by_var = pd.DataFrame(model_missing_by_var_rows)

    # --- Optional Excel writer (preserves original behaviour) ----------
    if excel_output_path:
        with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
            cohort_flow.to_excel(writer, sheet_name="cohort_flow", index=False)
            diagnostic_exclusions.to_excel(
                writer, sheet_name="diagnostic_exclusions", index=False
            )
            missing_data.to_excel(writer, sheet_name="missing_data", index=False)
            restoration_counts.to_excel(
                writer, sheet_name="restoration_counts", index=False
            )
            followup_checks.to_excel(writer, sheet_name="followup_checks", index=False)
            model_complete_case_table.to_excel(
                writer, sheet_name="model_complete_case", index=False
            )
            model_missing_by_var.to_excel(
                writer, sheet_name="model_missing_by_var", index=False
            )

    # --- Manuscript-ready text ----------------------------------------
    methods_cohort_text = (
        f"Methods - cohort construction: The analytical cohort comprised {rct_episode_n:,} primary root canal treatment episodes "
        f"identified from {raw_rows_n:,} source rows. After restricting to rows with non-missing Patient_ID, Tooth_Num, "
        f"Treatment_Date_dt, and Cohort ({raw_eligible_rows_n:,} rows), {pre_episode_combo_n:,} unique Patient_ID x Tooth_Num x Cohort "
        f"combinations were available before follow-up filtering. The final episode-level dataset contained {len(episodes):,} records, of which "
        f"{rct_episode_n:,} were primary root canal treatment episodes."
    )

    methods_missing_text = (
        f"Methods - missing-data handling: Missing-data assessment was performed on the analytical primary root canal treatment cohort, not on the full raw source table. "
        f"Core follow-up variables were audited within all {rct_episode_n:,} episodes, whereas failure_date was assessed conditionally among failures only "
        f"(n={failure_denominator:,}). Binary restoration indicators were treated as coded absence when equal to 0 and were not classified as missing. "
        f"For the multivariable complete-case audit, {model_complete_n:,} of {rct_denominator:,} episodes had complete data for all model variables available in the cohort using {model_basis}."
    )

    results_flow_text = (
        f"Results - participant/episode flow: The primary root canal treatment cohort included {rct_episode_n:,} episodes from {rct_patient_n:,} unique patients, "
        f"representing {rct_unique_patient_tooth_n:,} unique Patient_ID + Tooth_Num combinations. Among these episodes, {rct_failure_n:,} ended in extraction/failure "
        f"and {rct_censored_n:,} were censored/survived at the end of follow-up."
    )

    if model_vars_available:
        missing_statement_core = []
        for variable in [
            "Age",
            "AgeGroup",
            "Male",
            "Smoking",
            "Cancer",
            "Diabetes",
            "Hypertension",
            "Biphos_use",
        ]:
            if variable in missing_data["Variable"].values:
                row = missing_data.loc[missing_data["Variable"] == variable].iloc[0]
                if not pd.isna(row["Missing n"]):
                    missing_statement_core.append(
                        f"{variable}: {int(row['Missing n']):,} missing ({row['Missing %']:.2f}%)"
                    )
        missing_summary_text = (
            "; ".join(missing_statement_core)
            if missing_statement_core
            else "No optional descriptive/model columns were available for missing-data summarization."
        )
    else:
        missing_summary_text = (
            "Model variables were not available for complete-case auditing in the current cohort."
        )

    results_missing_text = (
        f"Results - missing-data statement: Within the primary root canal treatment cohort, missingness in the core survival variables was minimal or absent after analytical filtering. "
        f"Among failures, {failure_date_missing_in_failures_n if not pd.isna(failure_date_missing_in_failures_n) else '[X]'} episodes were missing failure_date; among censored episodes, "
        f"{stop_date_missing_in_censored_n if not pd.isna(stop_date_missing_in_censored_n) else '[X]'} were missing stop_date. Complete-case exclusion for the multivariable model affected "
        f"{model_excluded_n:,} of {rct_denominator:,} episodes. Variable-specific missingness was as follows: {missing_summary_text}"
    )

    return {
        "rct": rct_resolved,
        "rct_source_note": rct_source_note,
        "raw_prep_notes": raw_prep_notes,
        "rct_episode_n": rct_episode_n,
        "rct_patient_n": rct_patient_n,
        "rct_unique_patient_tooth_n": rct_unique_patient_tooth_n,
        "rct_failure_n": rct_failure_n,
        "rct_censored_n": rct_censored_n,
        "warnings": warnings,
        "cohort_flow": cohort_flow,
        "diagnostic_exclusions": diagnostic_exclusions,
        "missing_data": missing_data,
        "restoration_counts": restoration_counts,
        "followup_checks": followup_checks,
        "model_complete_case_table": model_complete_case_table,
        "model_missing_by_var": model_missing_by_var,
        "model_basis": model_basis,
        "model_vars_requested": model_vars_requested,
        "model_vars_available": model_vars_available,
        "methods_cohort_text": methods_cohort_text,
        "methods_missing_text": methods_missing_text,
        "results_flow_text": results_flow_text,
        "results_missing_text": results_missing_text,
        "excel_output_path": excel_output_path,
    }


# =============================================================================
# Cohort-description tables (Table 1 + Unified Screening Table)
# Mirrors the layout of analysis_utils.py from the implants research,
# adapted to the RCT episode dataset.
#
# Public API:
#   display_baseline_characteristics_table(rct, ...)
#   display_unified_screening_table(rct, ...)
#
# All math/statistics are local and self-contained; no notebook globals used.
# =============================================================================

import html as _html


# ---------------------------------------------------------------------------
# HTML rendering helpers (ported from analysis_utils.py, adapted column names)
# ---------------------------------------------------------------------------
def _display_html(html_text: str) -> None:
    """Render HTML in a Jupyter notebook; fall back to printing if IPython is absent."""
    try:
        import importlib
        html_cls = getattr(importlib.import_module("IPython.display"), "HTML")
        display_fn = getattr(importlib.import_module("IPython.display"), "display")
    except ImportError:
        print(html_text)
        return
    display_fn(html_cls(html_text))


def _render_table_html(
    dataframe: pd.DataFrame,
    title: Optional[str] = None,
    footnote: Optional[str] = None,
    section_rows: Optional[Iterable[int]] = None,
    detail_rows: Optional[Iterable[int]] = None,
    value_align: str = "right",
) -> str:
    """Two-column copyable table with optional section headers and indented details."""
    section_rows = set(section_rows or [])
    detail_rows = set(detail_rows or [])
    columns = list(dataframe.columns)

    wrapper_style = "max-width: 980px; margin: 0;"
    title_style = (
        "font-family: Calibri, Arial, sans-serif; font-size: 16px; font-weight: 700; "
        "margin: 0 0 10px 0; color: #111827;"
    )
    table_style = (
        "border-collapse: collapse; width: 100%; font-family: Calibri, Arial, sans-serif; "
        "font-size: 11pt; table-layout: fixed;"
    )
    header_cell_style = (
        "border-top: 1.5pt solid #374151; border-bottom: 1.5pt solid #374151; "
        "padding: 8px 10px; text-align: left; font-weight: 700; background-color: #ffffff; "
        "white-space: normal; word-break: break-word; overflow-wrap: anywhere;"
    )
    char_cell_style = (
        "padding: 7px 10px; border-bottom: 1px solid #d1d5db; vertical-align: top; "
        "text-align: left; white-space: normal; word-break: break-word; overflow-wrap: anywhere;"
    )
    value_cell_style = (
        "padding: 7px 10px; border-bottom: 1px solid #d1d5db; vertical-align: top; "
        f"text-align: {value_align}; white-space: normal; word-break: break-word; overflow-wrap: anywhere;"
    )
    section_char_style = (
        "padding: 8px 10px; border-top: 1.5pt solid #94a3b8; border-bottom: 1px solid #cbd5e1; "
        "background-color: #eef2f7; font-weight: 700; text-align: left; "
        "white-space: normal; word-break: break-word; overflow-wrap: anywhere;"
    )
    section_value_style = (
        "padding: 8px 10px; border-top: 1.5pt solid #94a3b8; border-bottom: 1px solid #cbd5e1; "
        "background-color: #eef2f7;"
    )
    footnote_style = (
        "font-family: Calibri, Arial, sans-serif; font-size: 10pt; margin-top: 8px; color: #374151;"
    )

    parts = [f'<div style="{wrapper_style}">']
    if title:
        parts.append(f'<div style="{title_style}">{_html.escape(title)}</div>')

    parts.append(f'<table style="{table_style}">')
    if len(columns) == 2:
        parts.append('<colgroup><col style="width: 62%;"><col style="width: 38%;"></colgroup>')
    parts.append("<thead><tr>")
    for col in columns:
        parts.append(f'<th style="{header_cell_style}">{_html.escape(str(col))}</th>')
    parts.append("</tr></thead><tbody>")

    for idx, row in dataframe.iterrows():
        parts.append("<tr>")
        is_section = idx in section_rows
        is_detail = idx in detail_rows

        for col_idx, col in enumerate(columns):
            cell_text = "" if pd.isna(row[col]) else str(row[col])
            if is_section:
                cell_style = section_char_style if col_idx == 0 else section_value_style
            else:
                cell_style = char_cell_style if col_idx == 0 else value_cell_style
                if col_idx == 0 and is_detail:
                    cell_style += " padding-left: 28px;"
            parts.append(f'<td style="{cell_style}">{_html.escape(cell_text)}</td>')
        parts.append("</tr>")

    parts.append("</tbody></table>")
    if footnote:
        parts.append(f'<div style="{footnote_style}">{_html.escape(footnote)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _render_wide_table_html(
    dataframe: pd.DataFrame,
    title: Optional[str] = None,
    footnote: Optional[str] = None,
    numeric_columns: Optional[Iterable[str]] = None,
) -> str:
    """Wide multi-column copyable table (used for the unified screening table)."""
    columns = list(dataframe.columns)
    numeric_columns = set(numeric_columns or [])

    wrapper_style = "max-width: 100%; margin: 0; overflow-x: auto;"
    title_style = (
        "font-family: Calibri, Arial, sans-serif; font-size: 16px; font-weight: 700; "
        "margin: 0 0 10px 0; color: #111827;"
    )
    table_style = (
        "border-collapse: collapse; width: 100%; min-width: 1380px; "
        "font-family: Calibri, Arial, sans-serif; font-size: 10.5pt; table-layout: auto;"
    )
    header_cell_style = (
        "border-top: 1.5pt solid #374151; border-bottom: 1.5pt solid #374151; "
        "padding: 8px 10px; text-align: left; font-weight: 700; background-color: #ffffff; "
        "white-space: normal; word-break: break-word; overflow-wrap: anywhere;"
    )
    text_cell_style = (
        "padding: 7px 10px; border-bottom: 1px solid #d1d5db; vertical-align: top; "
        "text-align: left; white-space: normal; word-break: break-word; overflow-wrap: anywhere;"
    )
    numeric_cell_style = (
        "padding: 7px 10px; border-bottom: 1px solid #d1d5db; vertical-align: top; "
        "text-align: right; white-space: normal; word-break: break-word; overflow-wrap: anywhere;"
    )
    footnote_style = (
        "font-family: Calibri, Arial, sans-serif; font-size: 10pt; margin-top: 8px; color: #374151;"
    )

    parts = [f'<div style="{wrapper_style}">']
    if title:
        parts.append(f'<div style="{title_style}">{_html.escape(title)}</div>')

    parts.append(f'<table style="{table_style}">')
    parts.append("<thead><tr>")
    for col in columns:
        parts.append(f'<th style="{header_cell_style}">{_html.escape(str(col))}</th>')
    parts.append("</tr></thead><tbody>")

    for _, row in dataframe.iterrows():
        parts.append("<tr>")
        for col in columns:
            cell_text = "" if pd.isna(row[col]) else str(row[col])
            cell_style = numeric_cell_style if col in numeric_columns else text_cell_style
            parts.append(f'<td style="{cell_style}">{_html.escape(cell_text)}</td>')
        parts.append("</tr>")

    parts.append("</tbody></table>")
    if footnote:
        parts.append(f'<div style="{footnote_style}">{_html.escape(footnote)}</div>')
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------
def _fmt_pct_str(value: float, decimals: int = 1, strip_trailing_zero: bool = False) -> str:
    formatted = f"{value:.{decimals}f}"
    if strip_trailing_zero and "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _fmt_count_pct(count: int, denominator: int, strip_trailing_zero: bool = False) -> str:
    pct = (count / denominator * 100) if denominator else 0.0
    return f"{count:,} ({_fmt_pct_str(pct, 1, strip_trailing_zero)}%)"


def _fmt_mean_sd_range(series: pd.Series, value_decimals: int = 1, range_decimals: int = 0) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "—"
    vfmt = f"{{:.{value_decimals}f}}"
    rfmt = f"{{:.{range_decimals}f}}"
    return (
        f"{vfmt.format(clean.mean())} ± {vfmt.format(clean.std())} "
        f"({rfmt.format(clean.min())}-{rfmt.format(clean.max())})"
    )


def _fmt_screening_pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.1f}"


def _fmt_screening_p(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _fmt_screening_hr(summary_row: Optional[pd.Series]) -> str:
    if summary_row is None:
        return "—"
    hr = summary_row.get("HR")
    hr_lo = summary_row.get("HR_lo")
    hr_hi = summary_row.get("HR_hi")
    if pd.isna(hr) or pd.isna(hr_lo) or pd.isna(hr_hi):
        return "—"
    return f"{hr:.2f} ({hr_lo:.2f}-{hr_hi:.2f})"


def _fmt_screening_num(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


# ---------------------------------------------------------------------------
# Follow-up summary on RCT episodes
# ---------------------------------------------------------------------------
def _rct_followup_summary(duration_days: pd.Series, event: pd.Series) -> Dict[str, float]:
    """Mean/SD/median in years, total tooth-years, incidence per 100 TY."""
    t = pd.to_numeric(pd.Series(duration_days), errors="coerce") / 365.25
    e = pd.to_numeric(pd.Series(event), errors="coerce").fillna(0).astype(int)
    mask = t.notna()
    t = t.loc[mask]
    e = e.loc[mask]
    n = int(len(t))
    if n == 0:
        return {
            "n": 0, "events": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
            "tooth_years": np.nan, "incidence_per_100_ty": np.nan,
        }
    tooth_years = float(t.sum())
    n_events = int(e.sum())
    incidence = (n_events / tooth_years * 100) if tooth_years > 0 else np.nan
    return {
        "n": n,
        "events": n_events,
        "mean": float(t.mean()),
        "std": float(t.std(ddof=1)) if n > 1 else 0.0,
        "median": float(t.median()),
        "tooth_years": tooth_years,
        "incidence_per_100_ty": incidence,
    }


# ---------------------------------------------------------------------------
# Configuration: variables shown in Table 1 (patient-level) and the screening table
# Reference categories follow the manuscript Methods (paragraph 43).
# ---------------------------------------------------------------------------
RCT_PATIENT_FLAG_LABELS: Dict[str, str] = {
    "Smoking": "Smoking",
    "Diabetes": "Diabetes mellitus",
    "Hypertension": "Hypertension",
    "Biphos_use": "Bisphosphonate use",
    "Cancer": "Malignancy",
}


def _series_sex(data: pd.DataFrame) -> pd.Series:
    """Derive Sex (Female/Male) from the binary Male flag."""
    return pd.Series(
        np.where(pd.to_numeric(data["Male"], errors="coerce").fillna(0) > 0, "Male", "Female"),
        index=data.index,
        dtype="object",
    )


def _series_arch(data: pd.DataFrame) -> pd.Series:
    """Derive Arch (Upper/Lower) from Tooth_Num via classify_tooth."""
    arch = data["Tooth_Num"].apply(lambda v: classify_tooth(v)[0])
    arch = arch.where(arch.isin(["Upper", "Lower"]), other=np.nan)
    return pd.Series(arch, index=data.index, dtype="object")


def _series_position(data: pd.DataFrame) -> pd.Series:
    """Derive Position (Anterior/Posterior) from Tooth_Num via classify_tooth."""
    pos = data["Tooth_Num"].apply(lambda v: classify_tooth(v)[1])
    pos = pos.where(pos.isin(["Anterior", "Posterior"]), other=np.nan)
    return pd.Series(pos, index=data.index, dtype="object")


RCT_SCREENING_VARIABLE_SPECS: List[Dict[str, Any]] = [
    {
        "label": "Age group",
        "column": "AgeGroup",
        "levels": ["<40", "40\u201360", "\u226560"],   # "<40", "40–60", "≥60"
        "reference": "40\u201360",
        "kind": "categorical",
    },
    {
        "label": "Sex",
        "kind": "derived",
        "levels": ["Female", "Male"],
        "reference": "Female",
        "series_fn": _series_sex,
    },
    {
        "label": "Smoking",
        "column": "Smoking",
        "levels": ["No", "Yes"],
        "reference": "No",
        "kind": "binary",
    },
    {
        "label": "Diabetes mellitus",
        "column": "Diabetes",
        "levels": ["No", "Yes"],
        "reference": "No",
        "kind": "binary",
    },
    {
        "label": "Hypertension",
        "column": "Hypertension",
        "levels": ["No", "Yes"],
        "reference": "No",
        "kind": "binary",
    },
    {
        "label": "Bisphosphonate use",
        "column": "Biphos_use",
        "levels": ["No", "Yes"],
        "reference": "No",
        "kind": "binary",
    },
    {
        "label": "Malignancy",
        "column": "Cancer",
        "levels": ["No", "Yes"],
        "reference": "No",
        "kind": "binary",
    },
    {
        "label": "Coronal restoration group",
        "column": "Coronal_Restoration_Group",
        "levels": ["Neither", "Sealing only", "Sealing + Crown"],
        "reference": "Neither",
        "kind": "categorical",
    },
    {
        "label": "Arch",
        "kind": "derived",
        "levels": ["Upper", "Lower"],
        "reference": "Upper",
        "series_fn": _series_arch,
    },
    {
        "label": "Tooth position",
        "kind": "derived",
        "levels": ["Anterior", "Posterior"],
        "reference": "Posterior",
        "series_fn": _series_position,
    },
]


# ---------------------------------------------------------------------------
# Per-variable screening: levels, log-rank, and Cox HR (cluster-robust)
# ---------------------------------------------------------------------------
def _prepare_screening_levels(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    if spec["kind"] == "derived":
        series = spec["series_fn"](df)
    else:
        series = df[spec["column"]]

    if spec["kind"] == "binary":
        series = pd.Series(
            np.where(pd.to_numeric(series, errors="coerce").fillna(0) > 0, "Yes", "No"),
            index=df.index,
            dtype="object",
        )
    else:
        series = pd.Series(series, index=df.index, dtype="object")

    out = df.copy()
    out["_screening_level"] = pd.Categorical(series, categories=spec["levels"], ordered=True)
    out["_time_years"] = pd.to_numeric(out["duration_days"], errors="coerce") / 365.25
    out = out[
        out["_screening_level"].notna()
        & out["_time_years"].notna()
        & out["event"].notna()
        & out["Patient_ID"].notna()
    ].copy()
    return out


def _summarize_screening_levels(data: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for level in data["_screening_level"].cat.categories:
        sub = data[data["_screening_level"] == level]
        n_total = int(len(sub))
        n_events = int(sub["event"].sum()) if n_total else 0
        ty = float(pd.to_numeric(sub["_time_years"], errors="coerce").sum()) if n_total else np.nan
        survival_pct = (1 - n_events / n_total) * 100 if n_total else np.nan
        incidence = (n_events / ty * 100) if ty and ty > 0 else np.nan
        rows.append({
            "level": str(level),
            "n": n_total,
            "events": n_events,
            "survival_pct": survival_pct,
            "tooth_years": ty,
            "incidence": incidence,
        })
    return rows


def _compute_screening_log_rank_p(data: pd.DataFrame) -> float:
    levels_obs = data["_screening_level"].dropna().astype(str)
    if levels_obs.nunique() < 2 or int(data["event"].sum()) == 0:
        return np.nan
    try:
        result = multivariate_logrank_test(
            data["_time_years"].values,
            levels_obs.values,
            data["event"].astype(int).values,
        )
        return float(result.p_value)
    except Exception:
        return np.nan


def _fit_screening_cox(data: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Per-variable Cox model with cluster-robust SE on Patient_ID. Returns
    a {level_label -> Series(HR, HR_lo, HR_hi, p)} mapping for non-reference levels."""
    categories = list(data["_screening_level"].cat.categories)
    reference = spec["reference"]
    design = pd.DataFrame(index=data.index)
    col_to_level: Dict[str, str] = {}

    for idx, level in enumerate(categories):
        if level == reference:
            continue
        cname = f"x_{idx}"
        design[cname] = (data["_screening_level"] == level).astype(float)
        col_to_level[cname] = level

    if design.empty:
        return {}

    # Drop constant or unstable columns (no variation, or all events / no events)
    stable = []
    for c in design.columns:
        exposed = design[c] == 1
        ref_mask = design[c] == 0
        if exposed.sum() == 0 or ref_mask.sum() == 0:
            continue
        ev_exposed = int(data.loc[exposed, "event"].sum())
        ev_ref = int(data.loc[ref_mask, "event"].sum())
        if ev_exposed in (0, int(exposed.sum())):
            continue
        if ev_ref in (0, int(ref_mask.sum())):
            continue
        stable.append(c)
    if not stable:
        return {}
    design = design[stable]

    fit_df = design.copy()
    fit_df["_time_years"] = data["_time_years"].values
    fit_df["event"] = data["event"].astype(int).values
    fit_df["Patient_ID"] = data["Patient_ID"].values

    try:
        model = CoxPHFitter()
        model.fit(
            fit_df,
            duration_col="_time_years",
            event_col="event",
            cluster_col="Patient_ID",
            robust=True,
        )
        summary = model.summary
    except Exception:
        return {}

    out: Dict[str, pd.Series] = {}
    for cname, level in col_to_level.items():
        if cname in summary.index:
            r = summary.loc[cname]
            out[level] = pd.Series({
                "HR": r.get("exp(coef)", np.nan),
                "HR_lo": r.get("exp(coef) lower 95%", np.nan),
                "HR_hi": r.get("exp(coef) upper 95%", np.nan),
                "p": r.get("p", np.nan),
            })
    return out


# ---------------------------------------------------------------------------
# Table 1 — Baseline characteristics
# ---------------------------------------------------------------------------
def make_baseline_characteristics_table(rct: pd.DataFrame) -> pd.DataFrame:
    """Table 1 for the primary RCT cohort. Layout mirrors the implants paper:
    cohort summary → patient-level → tooth-level → restoration-related."""
    value_col = "Value, n (%) unless otherwise specified"

    # Patient-level aggregation (one row per patient)
    patient_cols = ["Patient_ID", "Age", "Male"]
    for col in RCT_PATIENT_FLAG_LABELS:
        if col in rct.columns:
            patient_cols.append(col)
    keep = [c for c in patient_cols if c in rct.columns]
    pat_agg = {"Age": "first", "Male": "first"}
    for col in RCT_PATIENT_FLAG_LABELS:
        if col in rct.columns:
            pat_agg[col] = "max"
    pat_summary = rct[keep].groupby("Patient_ID", as_index=False).agg(pat_agg)

    n_patients = int(pat_summary.shape[0])
    n_episodes = int(len(rct))
    failures = int(pd.to_numeric(rct["event"], errors="coerce").fillna(0).sum())
    fu = _rct_followup_summary(rct["duration_days"], rct["event"])
    eps_per_pat = rct.groupby("Patient_ID").size()

    rows: List[Dict[str, Any]] = []

    # Cohort summary
    rows.append({"Characteristic": "Cohort summary", value_col: ""})
    rows.append({"Characteristic": "Patients, n", value_col: f"{n_patients:,}"})
    rows.append({"Characteristic": "Episodes analyzed, n", value_col: f"{n_episodes:,}"})
    rows.append({
        "Characteristic": "Episodes per patient (mean±SD, range)",
        value_col: _fmt_mean_sd_range(eps_per_pat, value_decimals=1, range_decimals=0),
    })
    rows.append({
        "Characteristic": "Overall tooth survival, %",
        value_col: _fmt_pct_str((1 - failures / n_episodes) * 100, 1) if n_episodes else "—",
    })
    rows.append({
        "Characteristic": "Tooth extractions, n (%)",
        value_col: _fmt_count_pct(failures, n_episodes),
    })
    rows.append({
        "Characteristic": "Follow-up time, years (mean±SD; median [IQR])",
        value_col: (
            f"{fu['mean']:.2f} ± {fu['std']:.2f}; "
            f"median {fu['median']:.2f} "
            f"[{pd.Series(rct['duration_days'].astype(float)/365.25).quantile(0.25):.2f}"
            f"\u2013"
            f"{pd.Series(rct['duration_days'].astype(float)/365.25).quantile(0.75):.2f}]"
        ),
    })
    rows.append({
        "Characteristic": "Total tooth-years of follow-up",
        value_col: f"{fu['tooth_years']:,.1f}",
    })
    rows.append({
        "Characteristic": "Failure incidence rate, per 100 tooth-years",
        value_col: f"{fu['incidence_per_100_ty']:.2f}",
    })

    # Patient-level characteristics
    rows.append({"Characteristic": "Patient-level characteristics", value_col: ""})
    rows.append({
        "Characteristic": "Age, years (mean±SD, range)",
        value_col: _fmt_mean_sd_range(pat_summary["Age"], value_decimals=1, range_decimals=0),
    })
    female = int((pd.to_numeric(pat_summary["Male"], errors="coerce").fillna(0) == 0).sum())
    male = int((pd.to_numeric(pat_summary["Male"], errors="coerce").fillna(0) == 1).sum())
    rows.append({"Characteristic": "Female", value_col: _fmt_count_pct(female, n_patients)})
    rows.append({"Characteristic": "Male", value_col: _fmt_count_pct(male, n_patients)})

    for col, label in RCT_PATIENT_FLAG_LABELS.items():
        if col in pat_summary.columns:
            count = int((pd.to_numeric(pat_summary[col], errors="coerce").fillna(0) > 0).sum())
            rows.append({"Characteristic": label, value_col: _fmt_count_pct(count, n_patients)})

    # Tooth-level characteristics
    arch = rct["Tooth_Num"].apply(lambda v: classify_tooth(v)[0])
    pos = rct["Tooth_Num"].apply(lambda v: classify_tooth(v)[1])

    rows.append({"Characteristic": "Tooth-level characteristics", value_col: ""})
    rows.append({"Characteristic": "Arch", value_col: ""})
    for label_in, label_out in [("Upper", "Upper jaw (maxilla)"), ("Lower", "Lower jaw (mandible)")]:
        c = int((arch == label_in).sum())
        rows.append({"Characteristic": label_out, value_col: _fmt_count_pct(c, n_episodes)})

    rows.append({"Characteristic": "Tooth position", value_col: ""})
    for label_in in ["Anterior", "Posterior"]:
        c = int((pos == label_in).sum())
        rows.append({"Characteristic": label_in, value_col: _fmt_count_pct(c, n_episodes)})

    # Restoration-related characteristics (coronal restoration group within window)
    rows.append({"Characteristic": "Restoration-related characteristics", value_col: ""})
    rows.append({"Characteristic": "Coronal restoration group", value_col: ""})
    if "Coronal_Restoration_Group" in rct.columns:
        for grp in ["Neither", "Sealing only", "Sealing + Crown"]:
            c = int((rct["Coronal_Restoration_Group"] == grp).sum())
            rows.append({"Characteristic": grp, value_col: _fmt_count_pct(c, n_episodes)})

    return pd.DataFrame(rows)


def display_baseline_characteristics_table(rct: pd.DataFrame) -> pd.DataFrame:
    """Render Table 1 (baseline characteristics) and return the underlying DataFrame."""
    table = make_baseline_characteristics_table(rct)
    value_col = [c for c in table.columns if c != "Characteristic"][0]

    # Section rows are blank-value rows; their following non-blank rows are details.
    section_mask = table[value_col].eq("")
    detail_mask: List[bool] = []
    in_section = False
    for is_section in section_mask.tolist():
        if is_section:
            in_section = True
            detail_mask.append(False)
        else:
            detail_mask.append(in_section)

    html_text = _render_table_html(
        table,
        title=(
            "Table 1. Baseline patient, tooth, and restoration characteristics "
            "of the primary nonsurgical RCT cohort"
        ),
        footnote=(
            "Values are presented as n (%) unless otherwise stated. Patient-level variables are "
            "reported per patient; tooth-level and restoration-related variables are reported per "
            "episode (one episode = one primary RCT on a given tooth). Follow-up is summarized in "
            "years as mean ± SD and median [IQR]. Incidence rate is reported per 100 tooth-years. "
            "Coronal restoration group reflects recorded sealing/crown procedures within the post-RCT "
            "window. SD, standard deviation; IQR, interquartile range."
        ),
        section_rows=table.index[section_mask].tolist(),
        detail_rows=table.index[[bool(v) for v in detail_mask]].tolist(),
        value_align="right",
    )
    _display_html(html_text)
    return table


# ---------------------------------------------------------------------------
# Unified Screening Table — per-variable N / events / survival / log-rank / Cox HR
# ---------------------------------------------------------------------------
_SCREENING_COLUMNS = [
    "Variable",
    "Level",
    "N",
    "Events",
    "Survival rate (%)",
    "Log-rank p-value",
    "Cox HR (95% CI)",
    "Cox p-value",
    "Tooth-years (TY)",
    "Failure incidence rate (per 100 TY)",
]
_SCREENING_NUMERIC = {
    "N", "Events", "Survival rate (%)",
    "Tooth-years (TY)", "Failure incidence rate (per 100 TY)",
}


def make_unified_screening_table(
    rct: pd.DataFrame,
    variable_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Build the per-variable screening table for the primary RCT cohort."""
    specs = variable_specs or RCT_SCREENING_VARIABLE_SPECS
    rows = []

    for spec in specs:
        analysis_df = _prepare_screening_levels(rct, spec)
        level_rows = _summarize_screening_levels(analysis_df)
        log_rank_p = _compute_screening_log_rank_p(analysis_df)
        cox_rows = _fit_screening_cox(analysis_df, spec)

        for idx, level_row in enumerate(level_rows):
            level = level_row["level"]
            cox_row = cox_rows.get(level)
            is_ref = level == spec["reference"]
            rows.append({
                "Variable": spec["label"] if idx == 0 else "",
                "Level": level,
                "N": f"{level_row['n']:,}",
                "Events": f"{level_row['events']:,}",
                "Survival rate (%)": _fmt_screening_pct(level_row["survival_pct"]),
                "Log-rank p-value": _fmt_screening_p(log_rank_p) if idx == 0 else "",
                "Cox HR (95% CI)": "Reference" if is_ref else _fmt_screening_hr(cox_row),
                "Cox p-value": "" if is_ref else (
                    _fmt_screening_p(cox_row["p"]) if cox_row is not None else "—"
                ),
                "Tooth-years (TY)": _fmt_screening_num(level_row["tooth_years"], decimals=1),
                "Failure incidence rate (per 100 TY)": _fmt_screening_num(
                    level_row["incidence"], decimals=2
                ),
            })

    return pd.DataFrame(rows, columns=_SCREENING_COLUMNS)


def display_unified_screening_table(
    rct: pd.DataFrame,
    variable_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Render the unified screening table and return the underlying DataFrame."""
    table = make_unified_screening_table(rct, variable_specs=variable_specs)
    html_text = _render_wide_table_html(
        table,
        title=(
            "Table 2. Unadjusted and per-variable adjusted survival outcomes "
            "(primary nonsurgical RCT cohort)"
        ),
        footnote=(
            "Tooth-years are calculated from each episode's follow-up duration "
            "(duration_days / 365.25) within the primary RCT cohort. Failure incidence rate is "
            "computed as Events / Tooth-years × 100. Log-rank p-values are descriptive and "
            "unadjusted. Cox hazard ratios are estimated per variable with cluster-robust standard "
            "errors on Patient_ID."
        ),
        numeric_columns=_SCREENING_NUMERIC,
    )
    _display_html(html_text)
    return table
