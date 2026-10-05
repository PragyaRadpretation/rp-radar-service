
import os
import json
import shutil
import tempfile
import threading
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.configuration.config import DICOM_TEMP_PATH, CONFIDENCE_THRESHOLD
from utils.utils import (
    check_orthanc_study_eligible,
    resolve_body_part_label,
    get_or_create_nifti,
    run_radar_inference,
    wipe_tmp_scratch
)

router = APIRouter()
os.makedirs(DICOM_TEMP_PATH, exist_ok=True)
gpu_lock = threading.Lock()


class InferencePayload(BaseModel):
    studyId: str
    pacsUrl: Optional[str] = None
    authCred: Optional[str] = None


@router.post("/ct_abdomen/predict")
def predict(payload: InferencePayload):
    try:
        pacs_url = payload.pacsUrl or os.getenv("PACS_URL", "https://pacs-ayurveda.radpretation.ai")
        auth_cred = payload.authCred or os.getenv("AUTH_CRED")
        auth_tuple = tuple(auth_cred.split(":", 1)) if auth_cred and ":" in auth_cred else None

        # 1. Eligibility Check
        is_eligible, orthanc_id, desc_text = check_orthanc_study_eligible(pacs_url, payload.studyId, auth_tuple)
        if not is_eligible:
            return JSONResponse(
                status_code=400,
                content={"error": f"Study rejected: {desc_text}"}
            )

        resolved_body_part = resolve_body_part_label(desc_text)

        # 2. Get/Convert NIfTI Volume (Cached)
        nii_path = get_or_create_nifti(pacs_url, orthanc_id, auth_tuple)

        # 3. Model Inference (GPU Locked)
        with gpu_lock:
            print(f"[*] Running RADAR inference on {orthanc_id}...")
            predictions_dict = run_radar_inference(nii_path)
            print(f"[✓] Inference complete for {orthanc_id}.")

        # 4. Filter findings >= 0.60 threshold
        abnormalities = [
            finding for finding, score in predictions_dict.items()
            if score >= CONFIDENCE_THRESHOLD
        ]
        print("results : ",abnormalities)
        response_data = {
            "normal": len(abnormalities) == 0,
            "abnormality": abnormalities,
            "body_part": resolved_body_part
        }

        # 5. Save Result JSON for backend retrieval
        temp_dir_return = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)
        json_filepath = os.path.join(temp_dir_return, "predictions.json")

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=4)

        return JSONResponse(content={
            "file_id": os.path.basename(temp_dir_return)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        wipe_tmp_scratch()


@router.get("/ct_abdomen/json/{file_id}")
async def get_json_object(file_id: str):
    path = os.path.join(DICOM_TEMP_PATH, file_id, "predictions.json")
    dir_path = os.path.join(DICOM_TEMP_PATH, file_id)

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="File not found")
    finally:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
