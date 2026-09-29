# import os
# import io
# import shutil
# import zipfile
# import tempfile
# import requests
# import dicom2nifti
# from pathlib import Path
# import pandas as pd
# from src.configuration.config import ABDOMEN_STUDY_REGEX, SUB_BODY_PARTS

# # Base workspace directory: ~/rp-radar-service
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Persistent cache directory: stores {study_id}.nii.gz
# NIFTI_CACHE_DIR = BASE_DIR / "nifti_cache"
# NIFTI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# # Temporary scratch space: stores transient DICOM downloads/conversions
# TMP_SCRATCH_DIR = BASE_DIR / "tmp" / "dicom_scratch"
# TMP_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# ORTHANC_URL = os.getenv("ORTHANC_URL", "http://localhost:8042")
# ORTHANC_USER = os.getenv("ORTHANC_USER", "orthanc")
# ORTHANC_PASS = os.getenv("ORTHANC_PASS", "orthanc")
# ORTHANC_AUTH = (ORTHANC_USER, ORTHANC_PASS) if ORTHANC_USER else None


# def resolve_orthanc_study(study_id: str) -> str:
#     """
#     Resolves a DICOM StudyInstanceUID or internal identifier to an Orthanc Study ID.
#     """
#     try:
#         res = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=ORTHANC_AUTH, timeout=10)
#         if res.status_code == 200:
#             return study_id

#         lookup_res = requests.post(f"{ORTHANC_URL}/tools/lookup", auth=ORTHANC_AUTH, json=[study_id], timeout=10).json()
#         for item in lookup_res:
#             if item.get("Type") == "Study":
#                 return item.get("ID")
#     except Exception as e:
#         print(f"[!] Error resolving study identifier '{study_id}': {e}")
#     return None


# def get_verified_abdomen_series(orthanc_study_id: str) -> str:
#     """
#     Selects the most suitable abdominal CT series within the study.
#     Validates modality (CT) and checks descriptions for abdominal keywords.
#     """
#     url = f"{ORTHANC_URL}/studies/{orthanc_study_id}/series"
#     series_list = requests.get(url, auth=ORTHANC_AUTH, timeout=10).json()

#     # Preferred match: modality is CT and description/body part indicates abdomen
#     for s in series_list:
#         tags = s.get("MainDicomTags", {})
#         modality = tags.get("Modality", "")
#         desc = tags.get("SeriesDescription", "").lower()
#         body_part = tags.get("BodyPartExamined", "").lower()

#         if modality == "CT" and any(k in desc or k in body_part for k in ["abd", "abdo", "kub", "pelvis"]):
#             return s["ID"]

#     # Fallback match: first available CT series in the study
#     for s in series_list:
#         if s.get("MainDicomTags", {}).get("Modality", "") == "CT":
#             return s["ID"]

#     return None


# def get_or_create_nifti(study_id: str) -> str:
#     """
#     Retrieves the cached NIfTI file or downloads and converts it from Orthanc.
#     Scratch DICOM files are deleted immediately after conversion.
#     """
#     cached_nii_path = NIFTI_CACHE_DIR / f"{study_id}.nii.gz"

#     # 1. Cache hit check
#     if cached_nii_path.exists() and cached_nii_path.stat().st_size > 0:
#         print(f"[*] Cache Hit: Using existing NIfTI at {cached_nii_path}")
#         return str(cached_nii_path)

#     # 2. Cache miss: verify study in Orthanc
#     print(f"[*] Cache Miss: Resolving study '{study_id}' in Orthanc...")
#     orthanc_study_id = resolve_orthanc_study(study_id)
#     if not orthanc_study_id:
#         raise ValueError(f"Study ID '{study_id}' could not be resolved in Orthanc.")

#     series_id = get_verified_abdomen_series(orthanc_study_id)
#     if not series_id:
#         raise ValueError(f"No suitable CT series found for study '{study_id}' in Orthanc.")

#     # 3. Download and convert using a self-destructing temporary directory
#     with tempfile.TemporaryDirectory(dir=str(TMP_SCRATCH_DIR)) as run_scratch:
#         dicom_dir = Path(run_scratch) / "dicoms"
#         nii_out_dir = Path(run_scratch) / "nifti_out"
#         dicom_dir.mkdir(parents=True, exist_ok=True)
#         nii_out_dir.mkdir(parents=True, exist_ok=True)

#         print(f"    -> Downloading DICOM archive for series '{series_id}'...")
#         archive_url = f"{ORTHANC_URL}/series/{series_id}/archive"
#         resp = requests.get(archive_url, auth=ORTHANC_AUTH, stream=True, timeout=120)
#         resp.raise_for_status()

#         with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
#             z.extractall(dicom_dir)

#         print(f"    -> Converting DICOM series to NIfTI format...")
#         dicom2nifti.convert_directory(str(dicom_dir), str(nii_out_dir), compression=True, reorient=True)

#         nii_candidates = list(nii_out_dir.glob("*.nii.gz"))
#         if not nii_candidates:
#             raise RuntimeError(f"dicom2nifti produced no .nii.gz files for series '{series_id}'.")

#         # Select primary reconstructed volume (ignoring scouts/localizers by file size)
#         primary_volume = max(nii_candidates, key=lambda f: f.stat().st_size)

#         # Move to persistent cache
#         shutil.move(str(primary_volume), str(cached_nii_path))

#     # All files inside run_scratch are automatically removed upon exiting the block
#     print(f"[✓] NIfTI conversion complete. Cached at: {cached_nii_path}")
#     return str(cached_nii_path)


# def wipe_tmp_scratch():
#     """
#     Utility function to clear any residual temporary artifacts inside tmp/.
#     """
#     if TMP_SCRATCH_DIR.exists():
#         for item in TMP_SCRATCH_DIR.iterdir():
#             if item.is_dir():
#                 shutil.rmtree(item, ignore_errors=True)
#             else:
#                 item.unlink(missing_ok=True)




# def check_orthanc_study_eligible(pacs_url: str, study_id: str, auth_tuple: tuple = None):
#     """Checks whether the study exists and has an abdominal CT series."""
#     try:
#         # Check direct study endpoint or lookup
#         res = requests.get(f"{pacs_url}/studies/{study_id}", auth=auth_tuple, timeout=10)
#         orthanc_id = study_id
#         if res.status_code != 200:
#             lookup = requests.post(f"{pacs_url}/tools/lookup", auth=auth_tuple, json=[study_id], timeout=10).json()
#             matches = [item["ID"] for item in lookup if item.get("Type") == "Study"]
#             if not matches:
#                 return False, None, "Study not found in Orthanc"
#             orthanc_id = matches[0]

#         # Inspect series descriptions
#         series_list = requests.get(f"{pacs_url}/studies/{orthanc_id}/series", auth=auth_tuple, timeout=10).json()
#         for s in series_list:
#             tags = s.get("MainDicomTags", {})
#             modality = tags.get("Modality", "")
#             desc = tags.get("SeriesDescription", "")
#             if modality == "CT" and ABDOMEN_STUDY_REGEX.search(desc):
#                 return True, orthanc_id, desc

#         return False, orthanc_id, "No matching abdominal CT series found"
#     except Exception as e:
#         return False, None, str(e)


# def resolve_body_part_label(desc_text: str) -> str:
#     """Resolves specific sub-anatomical body part or defaults to abdomen."""
#     for part, regex in SUB_BODY_PARTS.items():
#         if regex.search(desc_text):
#             return part
#     return "abdomen"


# def parse_radar_csv_output(csv_path: str, threshold: float = 0.60) -> list:
#     """Reads inference CSV and filters findings exceeding the confidence threshold."""
#     if not os.path.exists(csv_path):
#         return []
    
#     df = pd.read_csv(csv_path)
#     # Handles both (finding, probability) and wide-column CSV formats
#     abnormalities = []
#     if "finding" in df.columns and "probability" in df.columns:
#         filtered = df[df["probability"] >= threshold]
#         abnormalities = filtered["finding"].tolist()
#     else:
#         for col in df.columns:
#             if col.startswith("radar_") or col in df.columns:
#                 val = float(df[col].iloc[0])
#                 if val >= threshold:
#                     abnormalities.append(col.replace("radar_", ""))
#     return abnormalities


import os
import io
import shutil
import zipfile
import tempfile
import requests
import dicom2nifti
from pathlib import Path
from src.configuration.config import ABDOMEN_STUDY_REGEX, SUB_BODY_PARTS

BASE_DIR = Path(__file__).resolve().parent.parent
NIFTI_CACHE_DIR = BASE_DIR / "nifti_cache"
NIFTI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TMP_SCRATCH_DIR = BASE_DIR / "tmp" / "dicom_scratch"
TMP_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def check_orthanc_study_eligible(pacs_url: str, study_id: str, auth_tuple: tuple = None):
    """Verifies that the study exists and has an abdominal CT series."""
    try:
        res = requests.get(f"{pacs_url}/studies/{study_id}", auth=auth_tuple, timeout=10)
        orthanc_id = study_id
        if res.status_code != 200:
            lookup = requests.post(f"{pacs_url}/tools/lookup", auth=auth_tuple, json=[study_id], timeout=10).json()
            matches = [item["ID"] for item in lookup if item.get("Type") == "Study"]
            if not matches:
                return False, None, "Study not found in Orthanc"
            orthanc_id = matches[0]

        series_list = requests.get(f"{pacs_url}/studies/{orthanc_id}/series", auth=auth_tuple, timeout=10).json()
        for s in series_list:
            tags = s.get("MainDicomTags", {})
            modality = tags.get("Modality", "")
            desc = tags.get("SeriesDescription", "")
            body_part = tags.get("BodyPartExamined", "")
            combined_desc = f"{desc} {body_part}".strip()

            if modality == "CT" and ABDOMEN_STUDY_REGEX.search(combined_desc):
                return True, orthanc_id, combined_desc

        return False, orthanc_id, "No matching abdominal CT series found"
    except Exception as e:
        return False, None, str(e)


def resolve_body_part_label(desc_text: str) -> str:
    """Extracts sub-anatomical body part or defaults to abdomen."""
    for part, regex in SUB_BODY_PARTS.items():
        if regex.search(desc_text):
            return part
    return "abdomen"


def get_or_create_nifti(pacs_url: str, study_id: str, auth_tuple: tuple = None) -> str:
    """Fetches cached NIfTI or downloads DICOM from Orthanc and converts it."""
    cached_nii_path = NIFTI_CACHE_DIR / f"{study_id}.nii.gz"

    if cached_nii_path.exists() and cached_nii_path.stat().st_size > 0:
        print(f"[*] Cache Hit: Using {cached_nii_path}")
        return str(cached_nii_path)

    print(f"[*] Cache Miss: Downloading DICOMs for study {study_id}...")
    url = f"{pacs_url}/studies/{study_id}/series"
    series_list = requests.get(url, auth=auth_tuple, timeout=10).json()

    series_id = None
    for s in series_list:
        tags = s.get("MainDicomTags", {})
        if tags.get("Modality") == "CT":
            series_id = s["ID"]
            break

    if not series_id:
        raise ValueError(f"No CT series found in study {study_id}")

    with tempfile.TemporaryDirectory(dir=str(TMP_SCRATCH_DIR)) as run_scratch:
        dicom_dir = Path(run_scratch) / "dicoms"
        nii_out_dir = Path(run_scratch) / "nifti_out"
        dicom_dir.mkdir(parents=True, exist_ok=True)
        nii_out_dir.mkdir(parents=True, exist_ok=True)

        resp = requests.get(f"{pacs_url}/series/{series_id}/archive", auth=auth_tuple, stream=True, timeout=180)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(dicom_dir)

        dicom2nifti.convert_directory(str(dicom_dir), str(nii_out_dir), compression=True, reorient=True)
        nii_candidates = list(nii_out_dir.glob("*.nii.gz"))
        if not nii_candidates:
            raise RuntimeError(f"dicom2nifti produced no .nii.gz files for series '{series_id}'.")

        primary_volume = max(nii_candidates, key=lambda f: f.stat().st_size)
        shutil.move(str(primary_volume), str(cached_nii_path))

    return str(cached_nii_path)


def wipe_tmp_scratch():
    """Cleans temporary conversion files."""
    if TMP_SCRATCH_DIR.exists():
        for item in TMP_SCRATCH_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)