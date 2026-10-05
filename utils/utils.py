
import os
import io
import sys
import shutil
import zipfile
import tempfile
import requests
import pydicom
import dicom2nifti
import pandas as pd
import torch
from pathlib import Path
import dicom2nifti.settings as d2n_settings
import SimpleITK as sitk

from src.configuration.config import ABDOMEN_STUDY_REGEX, SUB_BODY_PARTS

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "ckpt"
RADAR_INFERENCE_DIR = BASE_DIR / "RADAR_inference"

# Ensure imports resolve
if str(RADAR_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(RADAR_INFERENCE_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Caching directories
NIFTI_CACHE_DIR = BASE_DIR / "nifti_cache"
NIFTI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TMP_SCRATCH_DIR = BASE_DIR / "tmp" / "dicom_scratch"
TMP_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Environmental variables required by RADAR
os.environ["MODEL_ROOT"] = str(CKPT_DIR)
os.environ["CONFIGS_ROOT"] = str(CKPT_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global model state holders for in-memory persistence
_PAD_FUNC = None
_MODEL = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. PACS & DICOM Processing
# ─────────────────────────────────────────────────────────────────────────────
def check_orthanc_study_eligible(pacs_url: str, study_id: str, auth_tuple: tuple = None):
    """
    Validates that:
    1. The study is an Abdominal CT (checks StudyDescription, SeriesDescription, and BodyPartExamined).
    2. A valid volumetric 3D CT series exists (>= 30 axial slices, non-scout).
    Rejects any non-abdomen studies explicitly.
    """
    try:
        pacs_root = pacs_url.rstrip("/")
        study_url = f"{pacs_root}/studies/{study_id}"
        res = requests.get(study_url, auth=auth_tuple, timeout=15)
        orthanc_id = study_id

        # 1. Resolve internal Orthanc ID if a DICOM StudyInstanceUID was passed
        if res.status_code == 404:
            find_resp = requests.post(
                f"{pacs_root}/tools/find",
                auth=auth_tuple,
                json={"Level": "Study", "Query": {"StudyInstanceUID": study_id}},
                timeout=15,
            )
            if find_resp.status_code == 200 and find_resp.json():
                orthanc_id = find_resp.json()[0]
                res = requests.get(f"{pacs_root}/studies/{orthanc_id}", auth=auth_tuple, timeout=15)
            else:
                return False, None, f"Study ID '{study_id}' not found in Orthanc"

        if res.status_code != 200:
            return False, None, f"Orthanc returned HTTP {res.status_code}"

        # 2. Check Study-Level Metadata (StudyDescription)
        study_data = res.json()
        study_tags = study_data.get("MainDicomTags", {})
        study_desc = (study_tags.get("StudyDescription") or "").strip()
        study_is_abdomen = bool(ABDOMEN_STUDY_REGEX.search(study_desc))

        # 3. Retrieve Series List
        series_resp = requests.get(f"{pacs_root}/studies/{orthanc_id}/series", auth=auth_tuple, timeout=15)
        if series_resp.status_code != 200:
            return False, orthanc_id, "Failed to retrieve series metadata from Orthanc"

        series_list = series_resp.json()
        candidate_series = []

        for s in series_list:
            tags = s.get("MainDicomTags", {})
            modality = tags.get("Modality", "")
            s_desc = (tags.get("SeriesDescription") or "").strip()
            body_part = (tags.get("BodyPartExamined") or "").strip()
            combined_series_text = f"{s_desc} {body_part}".strip()
            num_instances = len(s.get("Instances", []))

            s_desc_lower = s_desc.lower()
            is_scout = any(k in s_desc_lower for k in ["scout", "topogram", "localizer", "survey"])

            # Must be a 3D CT volume, not a scout/localizer
            if modality == "CT" and not is_scout and num_instances >= 30:
                series_is_abdomen = bool(ABDOMEN_STUDY_REGEX.search(combined_series_text))
                candidate_series.append({
                    "id": s["ID"],
                    "instances": num_instances,
                    "desc": combined_series_text,
                    "is_abdomen": series_is_abdomen
                })

        if not candidate_series:
            return False, orthanc_id, "No valid 3D CT volume (>= 30 slices) found in this study"

        # 4. Strict Abdomen Validation
        # Valid only if StudyDescription matches OR at least one CT series indicates abdomen/pelvis
        has_abdomen = study_is_abdomen or any(c["is_abdomen"] for c in candidate_series)

        if not has_abdomen:
            label = study_desc or candidate_series[0]["desc"] or "Unknown"
            return False, orthanc_id, f"This scan is not an Abdominal CT (Detected: '{label}')"

        # Pick the series with the most slices as primary volume
        candidate_series.sort(key=lambda x: x["instances"], reverse=True)
        primary_series = candidate_series[0]
        resolved_desc = study_desc or primary_series["desc"] or "CT Abdomen"

        return True, orthanc_id, resolved_desc

    except requests.exceptions.RequestException as e:
        return False, None, f"PACS network error: {str(e)}"
    except Exception as e:
        return False, None, f"Error validating study: {str(e)}"
    

def resolve_body_part_label(desc_text: str) -> str:
    """Extracts sub-anatomical body part or defaults to abdomen."""
    for part, regex in SUB_BODY_PARTS.items():
        if regex.search(desc_text):
            return part
    return "abdomen"
def get_or_create_nifti(pacs_url: str, study_id: str, auth_tuple: tuple = None) -> str:
    """
    Downloads DICOM slices from Orthanc, flattens them, and converts to NIfTI.
    Bypasses slice increment checks and uses SimpleITK fallback to handle irregular CT geometry.
    """
    cached_nii_path = NIFTI_CACHE_DIR / f"{study_id}.nii.gz"

    if cached_nii_path.exists() and cached_nii_path.stat().st_size > 0:
        print(f"[*] Cache Hit: Using {cached_nii_path}")
        return str(cached_nii_path)

    print(f"[*] Cache Miss: Resolving series for study {study_id}...")
    url = f"{pacs_url.rstrip('/')}/studies/{study_id}/series"
    series_resp = requests.get(url, auth=auth_tuple, timeout=15)
    series_resp.raise_for_status()
    series_list = series_resp.json()

    # Filter out scouts/localizers and non-CT series
    valid_series = []
    for s in series_list:
        tags = s.get("MainDicomTags", {})
        instances = s.get("Instances", [])
        num_instances = len(instances)
        desc = tags.get("SeriesDescription", "").lower()
        modality = tags.get("Modality", "")

        is_scout = any(k in desc for k in ["scout", "topogram", "localizer", "survey"])
        if modality == "CT" and not is_scout and num_instances >= 15:
            valid_series.append((s["ID"], num_instances, desc))

    if not valid_series:
        ct_candidates = [
            (s["ID"], len(s.get("Instances", [])), s.get("MainDicomTags", {}).get("SeriesDescription", ""))
            for s in series_list
            if s.get("MainDicomTags", {}).get("Modality", "") == "CT" and len(s.get("Instances", [])) >= 10
        ]
        if not ct_candidates:
            raise ValueError(f"No valid 3D CT volume found in study {study_id}")
        valid_series = ct_candidates

    # Pick the series with the highest slice count
    valid_series.sort(key=lambda x: x[1], reverse=True)
    selected_series_id, slice_count, desc = valid_series[0]
    print(f"[*] Selected series '{selected_series_id}' with {slice_count} slices.")

    with tempfile.TemporaryDirectory(dir=str(TMP_SCRATCH_DIR)) as run_scratch:
        extract_dir = Path(run_scratch) / "raw_extracted"
        flat_dicom_dir = Path(run_scratch) / "flat_dicoms"
        nii_out_dir = Path(run_scratch) / "nifti_out"

        extract_dir.mkdir(parents=True, exist_ok=True)
        flat_dicom_dir.mkdir(parents=True, exist_ok=True)
        nii_out_dir.mkdir(parents=True, exist_ok=True)

        resp = requests.get(f"{pacs_url.rstrip('/')}/series/{selected_series_id}/archive", auth=auth_tuple, stream=True, timeout=300)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(extract_dir)

        # Collect only valid DICOM slices into a flat folder
        valid_slices = 0
        for root, _, files in os.walk(extract_dir):
            for f in files:
                filepath = os.path.join(root, f)
                try:
                    pydicom.dcmread(filepath, stop_before_pixels=True)
                    shutil.copy(filepath, flat_dicom_dir / f"{valid_slices}_{f}")
                    valid_slices += 1
                except Exception:
                    continue

        if valid_slices < 10:
            raise ValueError(f"Only {valid_slices} DICOM slices extracted; minimum required is 10.")

        print(f"    -> Extracted {valid_slices} slices. Converting to NIfTI...")

        # 1. Primary conversion using dicom2nifti with relaxed validation
        d2n_settings.disable_validate_slice_increment()
        d2n_settings.disable_validate_orthogonal()
        d2n_settings.enable_resampling()
        d2n_settings.set_resample_spline_interpolation_order(1)

        conversion_successful = False
        try:
            dicom2nifti.convert_directory(str(flat_dicom_dir), str(nii_out_dir), compression=True, reorient=True)
            nii_candidates = list(nii_out_dir.glob("*.nii.gz"))
            if nii_candidates:
                primary_volume = max(nii_candidates, key=lambda f: f.stat().st_size)
                shutil.move(str(primary_volume), str(cached_nii_path))
                conversion_successful = True
                print(f"[✓] Converted via dicom2nifti: {cached_nii_path}")
        except Exception as e:
            print(f"[!] dicom2nifti conversion failed ({e}). Proceeding to SimpleITK fallback...")

        # 2. Resilient fallback using SimpleITK if dicom2nifti fails
        if not conversion_successful:
            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(flat_dicom_dir))
                if not dicom_names:
                    raise RuntimeError("SimpleITK found no valid DICOM series files in extracted directory.")
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                # Reorient to standard anatomical space (RAS)
                image = sitk.DICOMOrient(image, "RAS")
                sitk.WriteImage(image, str(cached_nii_path))
                print(f"[✓] Converted via SimpleITK: {cached_nii_path}")
            except Exception as sitk_err:
                raise RuntimeError(f"Both dicom2nifti and SimpleITK failed to convert study {study_id}: {sitk_err}")

    return str(cached_nii_path)

def wipe_tmp_scratch():
    """Cleans temporary conversion files."""
    if TMP_SCRATCH_DIR.exists():
        for item in TMP_SCRATCH_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. In-Memory RADAR Model Loading & Inference
# ─────────────────────────────────────────────────────────────────────────────
def warmup_radar_model():
    """Warms up and loads RADAR model into GPU memory once on startup."""
    global _PAD_FUNC, _MODEL
    if _MODEL is not None and _PAD_FUNC is not None:
        return _PAD_FUNC, _MODEL

    print("[*] Loading and warming up DAMO RADAR into GPU memory...")
    current_cwd = os.getcwd()
    try:
        os.chdir(str(RADAR_INFERENCE_DIR))
        from RADAR_inference.inference_demo import initialize
        _PAD_FUNC, _MODEL = initialize()
        print("[✓] DAMO RADAR ready in GPU memory.")
    finally:
        os.chdir(current_cwd)

    return _PAD_FUNC, _MODEL


def run_radar_inference(nii_path: str) -> dict:
    """
    Runs model inference on the NIfTI volume directly in memory.
    Returns:
        dict: {finding_name: probability_float}
    """
    pad_func, model = warmup_radar_model()

    from RADAR_inference.inference_demo import evaluate

    with tempfile.TemporaryDirectory() as temp_run_dir:
        input_dir = os.path.join(temp_run_dir, "input")
        save_dir = os.path.join(temp_run_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Symlink volume into RADAR's expected input directory
        symlink_name = os.path.basename(nii_path)
        symlink_path = os.path.join(input_dir, symlink_name)
        if not os.path.exists(symlink_path):
            os.symlink(os.path.abspath(nii_path), symlink_path)

        current_cwd = os.getcwd()
        try:
            os.chdir(str(RADAR_INFERENCE_DIR))
            evaluate(pad_func, model, input_dir, save_dir, save_tag="service")
        finally:
            os.chdir(current_cwd)

        csv_path = os.path.join(save_dir, "RADAR_infer_results_service.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"Model failed to generate results for volume: {nii_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            return {}

        predictions = {}
        first_row = df.iloc[0].to_dict()

        for col, val in first_row.items():
            if col == "file_name":
                continue
            # Extract clean English name from column header: "Chinese (English)"
            if "(" in col and ")" in col:
                finding_name = col.split("(")[-1].replace(")", "").strip()
            else:
                finding_name = col.strip()

            try:
                predictions[finding_name] = float(val) if val != "" and pd.notna(val) else 0.0
            except (ValueError, TypeError):
                continue

    return predictions