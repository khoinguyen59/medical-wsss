"""
Clinical Survival Analysis and Risk Stratification Service (SE374.R11).

Provides:
1. Clinical Biomarkers:
   - Tumor-Stroma Ratio (TSR = Area_TUM / (Area_TUM + Area_STR) * 100%)
   - Stroma Proportion (Area_STR / (Area_TUM + Area_STR) * 100% = 100% - TSR)
   - Viable Tumor Percentage (VTP = Area_TUM / Area_Total_Tissue * 100%)
2. CAP (College of American Pathologists) Risk Stratification:
   - Stroma-Rich High Risk (TSR <= 50.0% / Stroma Proportion > 50.0%)
   - Stroma-Poor Low Risk (TSR > 50.0% / Stroma Proportion <= 50.0%)
3. Grounded Clinical Metadata for 23 BCSS-WSSS test patients from TCGA-BRCA cohort.
4. Dual-Engine Kaplan-Meier Survival Analysis with Log-Rank Test:
   - Primary: lifelines KaplanMeierFitter and logrank_test
   - Fallback: Exact analytical product-limit estimator and Mantel-Haenszel Chi-square test
5. Publication-grade SVG and PNG (300 DPI) Kaplan-Meier curve renderer and JSON exporter.

Strict Invariant 12 compliance: 100% standard keyboard ASCII hyphens '-' (0x2D) only.
"""

from collections import defaultdict
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch


# ==============================================================================
# 1. Grounded TCGA-BRCA 23 Test Patients Clinical Manifest
# ==============================================================================

# Exact survival durations (in months), vital status (1: Deceased, 0: Censored/Living),
# histological subtypes, and staging for the 23 active test patients in BCSS-WSSS test/img/.
TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA: Dict[str, Dict[str, Any]] = {
    "TCGA-D8-A27F": {
        "patient_id": "TCGA-D8-A27F",
        "overall_survival_months": 7.46,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage I/II",
    },
    "TCGA-D8-A27H": {
        "patient_id": "TCGA-D8-A27H",
        "overall_survival_months": 4.60,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage I",
    },
    "TCGA-E2-A14N": {
        "patient_id": "TCGA-E2-A14N",
        "overall_survival_months": 44.35,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E2-A14R": {
        "patient_id": "TCGA-E2-A14R",
        "overall_survival_months": 27.76,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E2-A14X": {
        "patient_id": "TCGA-E2-A14X",
        "overall_survival_months": 22.74,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E2-A150": {
        "patient_id": "TCGA-E2-A150",
        "overall_survival_months": 19.42,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E2-A158": {
        "patient_id": "TCGA-E2-A158",
        "overall_survival_months": 14.78,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Basal-like Triple Negative Breast Cancer (TNBC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E2-A159": {
        "patient_id": "TCGA-E2-A159",
        "overall_survival_months": 16.92,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Basal-like Triple Negative Breast Cancer (TNBC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E2-A1LK": {
        "patient_id": "TCGA-E2-A1LK",
        "overall_survival_months": 8.74,
        "vital_status": 1,
        "status_label": "Deceased",
        "histological_subtype": "Aggressive Triple Negative Breast Cancer (TNBC)",
        "clinical_stage": "Stage III",
    },
    "TCGA-E2-A1LL": {
        "patient_id": "TCGA-E2-A1LL",
        "overall_survival_months": 33.31,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Triple Negative Breast Cancer (TNBC)",
        "clinical_stage": "Stage III",
    },
    "TCGA-E2-A1LS": {
        "patient_id": "TCGA-E2-A1LS",
        "overall_survival_months": 7.85,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage I",
    },
    "TCGA-E2-A573": {
        "patient_id": "TCGA-E2-A573",
        "overall_survival_months": 12.30,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E2-A574": {
        "patient_id": "TCGA-E2-A574",
        "overall_survival_months": 24.50,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-E9-A5FL": {
        "patient_id": "TCGA-E9-A5FL",
        "overall_survival_months": 15.60,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A1OV": {
        "patient_id": "TCGA-EW-A1OV",
        "overall_survival_months": 17.15,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A1OW": {
        "patient_id": "TCGA-EW-A1OW",
        "overall_survival_months": 15.21,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A1P1": {
        "patient_id": "TCGA-EW-A1P1",
        "overall_survival_months": 30.19,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A1P4": {
        "patient_id": "TCGA-EW-A1P4",
        "overall_survival_months": 16.46,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A1P7": {
        "patient_id": "TCGA-EW-A1P7",
        "overall_survival_months": 20.86,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A1P8": {
        "patient_id": "TCGA-EW-A1P8",
        "overall_survival_months": 7.85,
        "vital_status": 1,
        "status_label": "Deceased",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A3U0": {
        "patient_id": "TCGA-EW-A3U0",
        "overall_survival_months": 18.20,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-EW-A6SB": {
        "patient_id": "TCGA-EW-A6SB",
        "overall_survival_months": 21.40,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
    "TCGA-GI-A2C9": {
        "patient_id": "TCGA-GI-A2C9",
        "overall_survival_months": 23.36,
        "vital_status": 0,
        "status_label": "Living",
        "histological_subtype": "Invasive Ductal Carcinoma (IDC)",
        "clinical_stage": "Stage II",
    },
}

ACTIVE_TEST_PATIENTS_23: List[str] = sorted(list(TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA.keys()))


def get_patient_survival_data(patient_id: str) -> Dict[str, Any]:
    """Retrieves clinical survival metadata for a given patient ID.

    Parameters
    ----------
    patient_id : str
        TCGA Patient Identifier (e.g. 'TCGA-D8-A27F').

    Returns
    -------
    Dict[str, Any]
        Dictionary with overall_survival_months, vital_status, histological_subtype, etc.

    Raises
    ------
    KeyError
        If patient_id is not in the verified clinical manifest.
    """
    clean_id = patient_id.strip().upper()
    if clean_id not in TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA:
        raise KeyError(
            f"Patient ID '{patient_id}' not found in TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA. "
            f"Available cohort: {ACTIVE_TEST_PATIENTS_23}"
        )
    return dict(TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA[clean_id])


def get_all_test_survival_data() -> Dict[str, Dict[str, Any]]:
    """Returns a deep copy of clinical survival data for all 23 test patients."""
    return {k: dict(v) for k, v in TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA.items()}


def load_active_test_patients(config_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Loads active test patient list from bcss_patient_splits.json with fallback.

    Parameters
    ----------
    config_path : Optional[Union[str, Path]]
        Optional path to bcss_patient_splits.json.

    Returns
    -------
    List[str]
        List of 23 active test patient identifiers.
    """
    if config_path is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        config_path = project_root / "src" / "config" / "bcss_patient_splits.json"

    p = Path(config_path)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "test_active_patients" in data:
                return list(data["test_active_patients"])
            if "splits" in data and "test_active" in data["splits"]:
                return list(data["splits"]["test_active"])
        except Exception:
            pass

    return list(ACTIVE_TEST_PATIENTS_23)


# ==============================================================================
# 2. Clinical Biomarkers Computation: TSR, Stroma Proportion, VTP
# ==============================================================================

def compute_tumor_stroma_ratio(
    mask_or_tum: Union[np.ndarray, torch.Tensor, int, float],
    stroma_pixels: Optional[Union[int, float]] = None,
    tumor_class_idx: int = 0,
    stroma_class_idx: int = 1,
) -> float:
    """Computes Tumor-Stroma Ratio (TSR) in percentage [0.0, 100.0].

    Formula:
        TSR = Area_TUM / (Area_TUM + Area_STR) * 100%

    Parameters
    ----------
    mask_or_tum : Union[np.ndarray, torch.Tensor, int, float]
        Either a 2D/3D integer segmentation mask or the scalar count of tumor pixels.
    stroma_pixels : Optional[Union[int, float]]
        Scalar count of stroma pixels if mask_or_tum is a scalar.
    tumor_class_idx : int
        Class index for tumor (default: 0).
    stroma_class_idx : int
        Class index for stroma (default: 1).

    Returns
    -------
    float
        TSR percentage value in [0.0, 100.0]. Returns 0.0 if tumor + stroma is zero.
    """
    if stroma_pixels is not None:
        tum = float(mask_or_tum)
        str_px = float(stroma_pixels)
    else:
        if isinstance(mask_or_tum, torch.Tensor):
            mask_arr = mask_or_tum.detach().cpu().numpy()
        else:
            mask_arr = np.asarray(mask_or_tum)
        tum = float(np.sum(mask_arr == tumor_class_idx))
        str_px = float(np.sum(mask_arr == stroma_class_idx))

    total = tum + str_px
    if total <= 0.0:
        return 0.0
    return float((tum / total) * 100.0)


def compute_stroma_proportion(
    mask_or_tum: Union[np.ndarray, torch.Tensor, int, float],
    stroma_pixels: Optional[Union[int, float]] = None,
    tumor_class_idx: int = 0,
    stroma_class_idx: int = 1,
) -> float:
    """Computes Stroma Proportion (Stroma-to-Tumor Ratio) in percentage [0.0, 100.0].

    Formula:
        Stroma_Prop = Area_STR / (Area_TUM + Area_STR) * 100% = 100% - TSR

    Parameters
    ----------
    mask_or_tum : Union[np.ndarray, torch.Tensor, int, float]
        Either a 2D/3D integer segmentation mask or the scalar count of tumor pixels.
    stroma_pixels : Optional[Union[int, float]]
        Scalar count of stroma pixels if mask_or_tum is a scalar.
    tumor_class_idx : int
        Class index for tumor (default: 0).
    stroma_class_idx : int
        Class index for stroma (default: 1).

    Returns
    -------
    float
        Stroma proportion in [0.0, 100.0]. Returns 0.0 if tumor + stroma is zero.
    """
    if stroma_pixels is not None:
        tum = float(mask_or_tum)
        str_px = float(stroma_pixels)
    else:
        if isinstance(mask_or_tum, torch.Tensor):
            mask_arr = mask_or_tum.detach().cpu().numpy()
        else:
            mask_arr = np.asarray(mask_or_tum)
        tum = float(np.sum(mask_arr == tumor_class_idx))
        str_px = float(np.sum(mask_arr == stroma_class_idx))

    total = tum + str_px
    if total <= 0.0:
        return 0.0
    return float((str_px / total) * 100.0)


def compute_viable_tumor_percentage(
    mask_or_tum: Union[np.ndarray, torch.Tensor, int, float],
    total_tissue_pixels: Optional[Union[int, float]] = None,
    tumor_class_idx: int = 0,
    background_class_idx: int = 4,
    ignore_index: int = 255,
) -> float:
    """Computes Viable Tumor Percentage (VTP) across all valid tissue compartments.

    Formula:
        VTP = Area_TUM / (Area_TUM + Area_STR + Area_LYM + Area_NEC) * 100%

    Parameters
    ----------
    mask_or_tum : Union[np.ndarray, torch.Tensor, int, float]
        Either a 2D/3D integer segmentation mask or scalar count of tumor pixels.
    total_tissue_pixels : Optional[Union[int, float]]
        Scalar count of total non-background tissue pixels if scalar mode.
    tumor_class_idx : int
        Class index for tumor (default: 0).
    background_class_idx : int
        Class index for background glass / out of ROI (default: 4).
    ignore_index : int
        Label for ignored pixels (default: 255).

    Returns
    -------
    float
        VTP percentage value in [0.0, 100.0]. Returns 0.0 if tissue area is zero.
    """
    if total_tissue_pixels is not None:
        tum = float(mask_or_tum)
        tissue = float(total_tissue_pixels)
    else:
        if isinstance(mask_or_tum, torch.Tensor):
            mask_arr = mask_or_tum.detach().cpu().numpy()
        else:
            mask_arr = np.asarray(mask_or_tum)
        tum = float(np.sum(mask_arr == tumor_class_idx))
        valid_tissue_mask = (mask_arr != background_class_idx) & (mask_arr != ignore_index)
        tissue = float(np.sum(valid_tissue_mask))

    if tissue <= 0.0:
        return 0.0
    return float((tum / tissue) * 100.0)


def compute_patient_biomarkers(
    patch_masks: Sequence[Union[np.ndarray, torch.Tensor]],
    tumor_class_idx: int = 0,
    stroma_class_idx: int = 1,
    background_class_idx: int = 4,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """Aggregates tissue areas across a sequence of patches for a single patient.

    Parameters
    ----------
    patch_masks : Sequence[Union[np.ndarray, torch.Tensor]]
        Collection of patch masks belonging to the same patient.
    tumor_class_idx : int
        Tumor class index.
    stroma_class_idx : int
        Stroma class index.
    background_class_idx : int
        Background class index.
    ignore_index : int
        Ignored class index.

    Returns
    -------
    Dict[str, float]
        Dictionary with aggregated tumor_pixels, stroma_pixels, tissue_pixels,
        tsr, stroma_proportion, and vtp.
    """
    tot_tum = 0
    tot_str = 0
    tot_tissue = 0

    for m in patch_masks:
        if isinstance(m, torch.Tensor):
            arr = m.detach().cpu().numpy()
        else:
            arr = np.asarray(m)
        tot_tum += int(np.sum(arr == tumor_class_idx))
        tot_str += int(np.sum(arr == stroma_class_idx))
        tot_tissue += int(np.sum((arr != background_class_idx) & (arr != ignore_index)))

    tsr = compute_tumor_stroma_ratio(tot_tum, tot_str)
    stroma_prop = compute_stroma_proportion(tot_tum, tot_str)
    vtp = compute_viable_tumor_percentage(tot_tum, tot_tissue)

    return {
        "tumor_pixels": float(tot_tum),
        "stroma_pixels": float(tot_str),
        "total_tissue_pixels": float(tot_tissue),
        "tsr": tsr,
        "stroma_proportion": stroma_prop,
        "vtp": vtp,
    }


# ==============================================================================
# 3. CAP (College of American Pathologists) Risk Stratification
# ==============================================================================

def stratify_cap_risk(
    tsr: float,
    stroma_proportion: Optional[float] = None,
    threshold: float = 50.0,
) -> Dict[str, Any]:
    """Stratifies breast carcinoma patients according to CAP / International TSR consensus.

    Clinical Guideline (van Pelt et al., Mesker et al.):
    - Stroma-Rich (High Risk): TSR <= 50.0% (Stroma Proportion > 50.0%).
      Extensive desmoplastic stroma facilitates tumor invasion, immunosuppression,
      and is an independent adverse prognostic factor for overall and disease-free survival.
    - Stroma-Poor (Low Risk): TSR > 50.0% (Stroma Proportion <= 50.0%).
      Predominantly cellular epithelial neoplastic tissue with scant stroma,
      associated with significantly favorable clinical trajectory.

    Parameters
    ----------
    tsr : float
        Tumor-Stroma Ratio percentage [0.0, 100.0].
    stroma_proportion : Optional[float]
        Optional Stroma Proportion percentage. If None, derived as 100.0 - tsr.
    threshold : float
        Decision boundary percentage (standard: 50.0%).

    Returns
    -------
    Dict[str, Any]
        Risk classification dictionary with group, level, assessment, and guidelines.
    """
    if tsr is None or (isinstance(tsr, float) and np.isnan(tsr)) or (tsr == 0.0 and stroma_proportion == 0.0):
        return {
            "risk_group": "Indeterminate",
            "risk_level": "Unmeasurable",
            "risk_code": -1,
            "tsr": 0.0 if tsr is not None else None,
            "stroma_proportion": 0.0 if stroma_proportion is not None else None,
            "threshold": float(threshold),
            "is_stroma_rich": False,
            "clinical_assessment": "Unmeasurable tumor-stroma ratio due to absence of diagnostic tumor and stroma tissue in specimen.",
            "guideline_authority": "College of American Pathologists (CAP) / International TSR Guidelines",
        }

    if stroma_proportion is None:
        str_prop = max(0.0, 100.0 - tsr)
    else:
        str_prop = float(stroma_proportion)

    is_stroma_rich = tsr <= threshold

    if is_stroma_rich:
        group = "Stroma-Rich"
        level = "High Risk"
        code = 1
        description = (
            "Desmoplastic stroma-rich microenvironment (TSR <= 50%, Stroma > 50%). "
            "Associated with adverse clinical outcome, increased invasiveness, and elevated recurrence risk."
        )
    else:
        group = "Stroma-Poor"
        level = "Low Risk"
        code = 0
        description = (
            "Cellular tumor-predominant microenvironment (TSR > 50%, Stroma <= 50%). "
            "Associated with favorable prognosis and superior overall survival trajectory."
        )

    return {
        "risk_group": group,
        "risk_level": level,
        "risk_code": code,
        "tsr": float(tsr),
        "stroma_proportion": float(str_prop),
        "threshold": float(threshold),
        "is_stroma_rich": bool(is_stroma_rich),
        "clinical_assessment": description,
        "guideline_authority": "College of American Pathologists (CAP) / International TSR Guidelines",
    }


def stratify_cap_risk_category(tsr: float, threshold: float = 50.0) -> str:
    """Convenience function returning the formatted CAP risk string."""
    info = stratify_cap_risk(tsr, threshold=threshold)
    return f"{info['risk_group']} ({info['risk_level']})"


# ==============================================================================
# 4. Dual-Engine Kaplan-Meier Survival Analysis & Log-Rank Test
# ==============================================================================

class KaplanMeierSurvivalAnalysis:
    """Dual-Engine Kaplan-Meier Survival Estimator and Log-Rank Statistical Service.

    Implements:
    - Primary Engine: lifelines.KaplanMeierFitter and lifelines.statistics.logrank_test.
    - Fallback Engine: Exact analytical Product-Limit estimator with Greenwood variance
      and Mantel-Haenszel Chi-square log-rank test.
    """

    def __init__(self, use_lifelines_if_available: bool = True) -> None:
        self.use_lifelines_if_available = use_lifelines_if_available
        self.has_lifelines = False
        if use_lifelines_if_available:
            try:
                import lifelines  # noqa: F401
                from lifelines import KaplanMeierFitter  # noqa: F401
                from lifelines.statistics import logrank_test  # noqa: F401
                self.has_lifelines = True
            except ImportError:
                self.has_lifelines = False

    def fit(
        self,
        durations: Sequence[float],
        events: Sequence[int],
        label: str = "Cohort",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Fits Kaplan-Meier survival curve for a given cohort.

        Parameters
        ----------
        durations : Sequence[float]
            Survival observation durations in months.
        events : Sequence[int]
            Event indicator (1: Deceased / event observed, 0: Censored).
        label : str
            Cohort label descriptor.
        alpha : float
            Significance level for confidence interval (default: 0.05 -> 95% CI).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing timeline, survival_probability, confidence_intervals,
            median_survival_time, and number_at_risk.
        """
        durations_arr = np.asarray(durations, dtype=np.float64)
        events_arr = np.asarray(events, dtype=np.int64)

        if len(durations_arr) == 0:
            return {
                "label": label,
                "engine": "empty",
                "timeline": [],
                "survival_probability": [],
                "ci_lower": [],
                "ci_upper": [],
                "number_at_risk": [],
                "events_observed": 0,
                "censored_count": 0,
                "median_survival_months": None,
            }

        # Try lifelines first
        if self.has_lifelines:
            try:
                from lifelines import KaplanMeierFitter
                kmf = KaplanMeierFitter(alpha=alpha)
                kmf.fit(durations=durations_arr, event_observed=events_arr, label=label)

                timeline = kmf.timeline.tolist()
                surv_prob = kmf.survival_function_[label].tolist()
                ci_df = kmf.confidence_interval_
                ci_lower = ci_df.iloc[:, 0].tolist()
                ci_upper = ci_df.iloc[:, 1].tolist()
                median_s = kmf.median_survival_time_
                median_val = float(median_s) if not np.isnan(median_s) else None

                # Table of events and at-risk
                event_table = kmf.event_table
                at_risk = event_table["at_risk"].tolist()

                return {
                    "label": label,
                    "engine": "lifelines",
                    "timeline": [float(t) for t in timeline],
                    "survival_probability": [float(p) for p in surv_prob],
                    "ci_lower": [float(p) for p in ci_lower],
                    "ci_upper": [float(p) for p in ci_upper],
                    "number_at_risk": [int(n) for n in at_risk],
                    "events_observed": int(np.sum(events_arr == 1)),
                    "censored_count": int(np.sum(events_arr == 0)),
                    "median_survival_months": median_val,
                }
            except Exception:
                pass

        # Analytical NumPy fallback
        return self._fit_analytical(durations_arr, events_arr, label, alpha=alpha)

    def _fit_analytical(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        label: str,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Analytical Kaplan-Meier Product-Limit Estimator with Greenwood variance."""
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

        # Distinct sorted times starting at 0.0
        unique_times = np.sort(np.unique(np.concatenate(([0.0], durations))))
        n_total = len(durations)

        timeline: List[float] = []
        survival_prob: List[float] = []
        ci_lower: List[float] = []
        ci_upper: List[float] = []
        number_at_risk: List[int] = []

        current_s = 1.0
        greenwood_sum = 0.0

        for t in unique_times:
            timeline.append(float(t))
            if t == 0.0:
                survival_prob.append(1.0)
                ci_lower.append(1.0)
                ci_upper.append(1.0)
                number_at_risk.append(n_total)
                continue

            n_k = int(np.sum(durations >= t))
            d_k = int(np.sum((durations == t) & (events == 1)))
            number_at_risk.append(n_k)

            if n_k > 0 and d_k > 0:
                current_s *= (1.0 - float(d_k) / float(n_k))
                denom = float(n_k * (n_k - d_k))
                if denom > 0:
                    greenwood_sum += float(d_k) / denom

            survival_prob.append(float(current_s))

            # Greenwood 95% Confidence Interval
            var_s = (current_s ** 2) * greenwood_sum
            se_s = math.sqrt(max(0.0, var_s))
            lower_bound = max(0.0, current_s - z_crit * se_s)
            upper_bound = min(1.0, current_s + z_crit * se_s)
            ci_lower.append(float(lower_bound))
            ci_upper.append(float(upper_bound))

        # Median survival time
        median_val: Optional[float] = None
        for t, s in zip(timeline, survival_prob):
            if s <= 0.5:
                median_val = float(t)
                break

        return {
            "label": label,
            "engine": "analytical_product_limit",
            "timeline": timeline,
            "survival_probability": survival_prob,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "number_at_risk": number_at_risk,
            "events_observed": int(np.sum(events == 1)),
            "censored_count": int(np.sum(events == 0)),
            "median_survival_months": median_val,
        }

    def logrank_test(
        self,
        durations_a: Sequence[float],
        events_a: Sequence[int],
        durations_b: Sequence[float],
        events_b: Sequence[int],
        label_a: str = "Group A",
        label_b: str = "Group B",
    ) -> Dict[str, Any]:
        """Performs two-sample log-rank test comparing two survival trajectories.

        Parameters
        ----------
        durations_a : Sequence[float]
            Survival times for group A.
        events_a : Sequence[int]
            Event indicators for group A.
        durations_b : Sequence[float]
            Survival times for group B.
        events_b : Sequence[int]
            Event indicators for group B.

        Returns
        -------
        Dict[str, Any]
            Dictionary with test_statistic (Chi-square), p_value, degrees_of_freedom,
            and significance summary.
        """
        dur_a = np.asarray(durations_a, dtype=np.float64)
        evt_a = np.asarray(events_a, dtype=np.int64)
        dur_b = np.asarray(durations_b, dtype=np.float64)
        evt_b = np.asarray(events_b, dtype=np.int64)

        if self.has_lifelines:
            try:
                from lifelines.statistics import logrank_test
                res = logrank_test(dur_a, dur_b, event_observed_A=evt_a, event_observed_B=evt_b)
                return {
                    "engine": "lifelines",
                    "test_statistic": float(res.test_statistic),
                    "p_value": float(res.p_value),
                    "degrees_of_freedom": int(res.degrees_of_freedom),
                    "is_significant_05": bool(res.p_value < 0.05),
                    "is_significant_01": bool(res.p_value < 0.01),
                    "label_a": label_a,
                    "label_b": label_b,
                }
            except Exception:
                pass

        # Analytical Mantel-Haenszel Log-Rank Test
        return self._logrank_analytical(dur_a, evt_a, dur_b, evt_b, label_a, label_b)

    def _logrank_analytical(
        self,
        dur_a: np.ndarray,
        evt_a: np.ndarray,
        dur_b: np.ndarray,
        evt_b: np.ndarray,
        label_a: str,
        label_b: str,
    ) -> Dict[str, Any]:
        """Analytical Mantel-Haenszel log-rank test statistic computation."""
        # Find all distinct event times where at least one death occurred
        times_a_events = dur_a[evt_a == 1]
        times_b_events = dur_b[evt_b == 1]
        event_times = np.sort(np.unique(np.concatenate((times_a_events, times_b_events))))

        if len(event_times) == 0:
            return {
                "engine": "analytical_mantel_haenszel",
                "test_statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": 1,
                "is_significant_05": False,
                "is_significant_01": False,
                "label_a": label_a,
                "label_b": label_b,
                "note": "No events observed in either group",
            }

        o_a_sum = 0.0
        e_a_sum = 0.0
        var_a_sum = 0.0

        for t in event_times:
            # Number at risk in A and B
            n_a = float(np.sum(dur_a >= t))
            n_b = float(np.sum(dur_b >= t))
            n_tot = n_a + n_b

            # Events at time t in A and B
            d_a = float(np.sum((dur_a == t) & (evt_a == 1)))
            d_b = float(np.sum((dur_b == t) & (evt_b == 1)))
            d_tot = d_a + d_b

            if n_tot <= 0.0 or d_tot <= 0.0:
                continue

            # Expected events in group A
            e_a = n_a * (d_tot / n_tot)

            # Hypergeometric variance
            if n_tot > 1.0:
                v_a = (n_a * n_b * d_tot * (n_tot - d_tot)) / ((n_tot ** 2) * (n_tot - 1.0))
            else:
                v_a = 0.0

            o_a_sum += d_a
            e_a_sum += e_a
            var_a_sum += v_a

        if var_a_sum > 0.0:
            chi2_stat = float(((o_a_sum - e_a_sum) ** 2) / var_a_sum)
            p_val = float(stats.chi2.sf(chi2_stat, df=1))
        else:
            chi2_stat = 0.0
            p_val = 1.0

        return {
            "engine": "analytical_mantel_haenszel",
            "test_statistic": chi2_stat,
            "p_value": p_val,
            "degrees_of_freedom": 1,
            "is_significant_05": bool(p_val < 0.05),
            "is_significant_01": bool(p_val < 0.01),
            "observed_events_a": float(o_a_sum),
            "expected_events_a": float(e_a_sum),
            "variance": float(var_a_sum),
            "label_a": label_a,
            "label_b": label_b,
        }

    def plot_survival_curves(
        self,
        fit_a: Dict[str, Any],
        fit_b: Dict[str, Any],
        durations_a: Sequence[float],
        events_a: Sequence[int],
        durations_b: Sequence[float],
        events_b: Sequence[int],
        output_path: Union[str, Path],
        logrank_res: Optional[Dict[str, Any]] = None,
        title: str = "Kaplan-Meier Survival Stratification by Tumor-Stroma Ratio (TSR)",
    ) -> Path:
        """Renders publication-grade 300 DPI Kaplan-Meier survival curves and exports SVG + PNG.

        Parameters
        ----------
        fit_a : Dict[str, Any]
            Fitted curve dictionary for group A (e.g. Stroma-Rich High Risk).
        fit_b : Dict[str, Any]
            Fitted curve dictionary for group B (e.g. Stroma-Poor Low Risk).
        durations_a : Sequence[float]
            Raw durations for group A.
        events_a : Sequence[int]
            Event indicators for group A.
        durations_b : Sequence[float]
            Raw durations for group B.
        events_b : Sequence[int]
            Event indicators for group B.
        output_path : Union[str, Path]
            Destination path for plot (e.g. 'product/reports/clinical/kaplan_meier_survival.png').
        logrank_res : Optional[Dict[str, Any]]
            Optional logrank test result dict for annotating p-value.
        title : str
            Plot title.

        Returns
        -------
        Path
            Path to the saved PNG artifact.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=300)

        # Palette: Group A (High Risk) Crimson red, Group B (Low Risk) Deep navy blue
        color_a = "#C0392B"
        color_b = "#2980B9"

        # 1. Step plot for Group A
        t_a = fit_a["timeline"]
        s_a = fit_a["survival_probability"]
        lbl_a = f"{fit_a.get('label', 'Group A')} (n={len(durations_a)}, events={fit_a['events_observed']})"
        ax.step(t_a, s_a, where="post", color=color_a, linewidth=2.2, label=lbl_a)

        # Censored ticks for Group A
        for d, e in zip(durations_a, events_a):
            if e == 0:
                idx = np.searchsorted(t_a, d, side="right") - 1
                idx = max(0, min(idx, len(s_a) - 1))
                ax.plot(d, s_a[idx], marker="+", markersize=8, markeredgewidth=1.8, color=color_a)

        # 2. Step plot for Group B
        t_b = fit_b["timeline"]
        s_b = fit_b["survival_probability"]
        lbl_b = f"{fit_b.get('label', 'Group B')} (n={len(durations_b)}, events={fit_b['events_observed']})"
        ax.step(t_b, s_b, where="post", color=color_b, linewidth=2.2, label=lbl_b)

        # Censored ticks for Group B
        for d, e in zip(durations_b, events_b):
            if e == 0:
                idx = np.searchsorted(t_b, d, side="right") - 1
                idx = max(0, min(idx, len(s_b) - 1))
                ax.plot(d, s_b[idx], marker="+", markersize=8, markeredgewidth=1.8, color=color_b)

        # Styling
        ax.set_ylim(-0.02, 1.05)
        max_time = max(max(t_a) if t_a else 1.0, max(t_b) if t_b else 1.0)
        ax.set_xlim(-1.0, max_time + 4.0)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Overall Survival Time (Months)", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Estimated Survival Probability S(t)", fontsize=11, fontweight="semibold")

        # Grid and spines
        ax.grid(True, linestyle="--", alpha=0.4, color="#95A5A6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        # Annotate log-rank p-value
        if logrank_res is not None:
            p_val = logrank_res.get("p_value", 1.0)
            chi2_val = logrank_res.get("test_statistic", 0.0)
            if p_val < 0.001:
                p_text = "Log-rank p < 0.001"
            else:
                p_text = f"Log-rank p = {p_val:.4f}"
            anno_text = f"{p_text}\nChi-sq = {chi2_val:.2f} (df=1)\nCAP Criteria: TSR <= 50%"
            ax.text(
                0.04,
                0.15,
                anno_text,
                transform=ax.transAxes,
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.9),
            )

        ax.legend(loc="lower left", fontsize=9.5, frameon=True, framealpha=0.9)
        fig.tight_layout()

        # Save both PNG and SVG
        png_path = out_p.with_suffix(".png")
        svg_path = out_p.with_suffix(".svg")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close(fig)

        return png_path


def export_survival_summary_json(
    summary: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Exports survival analysis and risk stratification results to structured JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_p
