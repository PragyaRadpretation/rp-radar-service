# import os
# import sys

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# RADAR_INFERENCE_DIR = os.path.join(BASE_DIR, "RADAR_inference")

# if RADAR_INFERENCE_DIR not in sys.path:
#     sys.path.insert(0, RADAR_INFERENCE_DIR)
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)

# from RADAR_inference.inference_demo import run_model

# def run_radar_inference(nii_path: str) -> dict:
#     """
#     Runs RADAR forward pass directly on the NIfTI volume.
#     Returns: dict of {finding_name: probability}
#     """
#     results_df = run_model(nii_path)
#     return dict(zip(results_df["finding"], results_df["probability"].astype(float)))


import os
import sys
import tempfile
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
RADAR_INFERENCE_DIR = os.path.join(BASE_DIR, "RADAR_inference")

# Ensure required search paths and environment variables
if RADAR_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, RADAR_INFERENCE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["MODEL_ROOT"] = CKPT_DIR
os.environ["CONFIGS_ROOT"] = CKPT_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_radar_inference(nii_path: str) -> dict:
    """
    Executes single-volume RADAR inference using infer_single.py,
    reads output probabilities into memory, and cleans up the temporary CSV.
    """
    infer_script = os.path.join(RADAR_INFERENCE_DIR, "infer_single.py")
    if not os.path.exists(infer_script):
        raise FileNotFoundError(f"Inference script missing: {infer_script}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_csv = os.path.join(tmp_dir, "radar_preds.csv")

        # Execute infer_single.py
        cmd = (
            f"python '{infer_script}' "
            f"--img_path '{nii_path}' "
            f"--output_csv '{output_csv}'"
        )
        ret = os.system(cmd)
        if ret != 0 or not os.path.exists(output_csv):
            raise RuntimeError(f"RADAR inference process crashed or failed with exit code: {ret}")

        # Parse CSV output into a clean dictionary
        df = pd.read_csv(output_csv)
        if df.empty:
            return {}

        predictions = {}
        # Case A: Standard tall format (finding, probability)
        if "finding" in df.columns and "probability" in df.columns:
            for _, row in df.iterrows():
                predictions[str(row["finding"]).strip()] = float(row["probability"])
        # Case B: Wide tabular format (each finding is a column name)
        else:
            row = df.iloc[0].to_dict()
            for col, val in row.items():
                if col not in ["file_name", "index", "id"]:
                    clean_finding = col.split("_(")[0].strip()
                    try:
                        predictions[clean_finding] = float(val)
                    except (ValueError, TypeError):
                        continue

        return predictions