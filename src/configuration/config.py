# import os
# import re
# from pathlib import Path          # <--- ADD THIS
# from dotenv import load_dotenv

# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# load_dotenv(BASE_DIR / ".env")

# PACS_URL = os.getenv("PACS_URL", "https://pacs-ayurveda.radpretation.ai")
# AUTH_CRED = os.getenv("AUTH_CRED", "orthanc:orthanc")
# CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
# DICOM_TEMP_PATH = str(BASE_DIR / "tmp" / "dicom_uploads")

# # Fast verification pattern for Abdominal / Pelvic / KUB scans
# ABDOMEN_STUDY_REGEX = re.compile(
#     r"\b(abd|abdo|abdomen|kub|pelvis|pelvic|hip|hips|w/a|cect\s*ab|ncct\s*ab|liver|renal|kidney)\b"
#     r"|(\bct\b.*\b(abd|abdo|abdomen|kub|pelvis|hip)\b)"
#     r"|(\b(chest\s*\+?\s*abdomen|abdomen\s*\+?\s*pelvis)\b)",
#     re.IGNORECASE
# )

# # Sub-anatomical resolution logic for the "body_part" field
# SUB_BODY_PARTS = {
#     "hip": re.compile(r"\b(hip|hips)\b", re.IGNORECASE),
#     "pelvis": re.compile(r"\b(pelvis|pelvic)\b", re.IGNORECASE),
#     "kub": re.compile(r"\b(kub|kidney|ureter|bladder|renal)\b", re.IGNORECASE),
#     "liver": re.compile(r"\b(liver|hepatic)\b", re.IGNORECASE)
# }

import os
import re
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

PACS_URL = os.getenv("PACS_URL", "https://pacs-ayurveda.radpretation.ai")
AUTH_CRED = os.getenv("AUTH_CRED", "orthanc:orthanc")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
DICOM_TEMP_PATH = str(BASE_DIR / "tmp" / "dicom_uploads")

ABDOMEN_STUDY_REGEX = re.compile(
    r"\b(abd|abdo|abdomen|kub|pelvis|pelvic|hip|hips|w/a|cect\s*ab|ncct\s*ab|liver|renal|kidney)\b"
    r"|(\bct\b.*\b(abd|abdo|abdomen|kub|pelvis|hip)\b)"
    r"|(\b(chest\s*\+?\s*abdomen|abdomen\s*\+?\s*pelvis)\b)",
    re.IGNORECASE
)

SUB_BODY_PARTS = {
    "hip": re.compile(r"\b(hip|hips)\b", re.IGNORECASE),
    "pelvis": re.compile(r"\b(pelvis|pelvic)\b", re.IGNORECASE),
    "kub": re.compile(r"\b(kub|kidney|ureter|bladder|renal)\b", re.IGNORECASE),
    "liver": re.compile(r"\b(liver|hepatic)\b", re.IGNORECASE)
}