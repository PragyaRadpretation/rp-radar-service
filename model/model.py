
import os
import sys
import shutil
import tempfile
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
RADAR_INFERENCE_DIR = os.path.join(BASE_DIR, "RADAR_inference")

if RADAR_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, RADAR_INFERENCE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Configure required environments
os.environ["MODEL_ROOT"] = CKPT_DIR
os.environ["CONFIGS_ROOT"] = CKPT_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global model state holders for warm-up
_PAD_FUNC = None
_MODEL = None


def warmup_radar_model():
    """Warms up and keeps the RADAR model in GPU memory on service launch."""
    global _PAD_FUNC, _MODEL
    if _MODEL is not None and _PAD_FUNC is not None:
        return _PAD_FUNC, _MODEL

    print("[*] Warming up DAMO RADAR model into GPU memory...")
    # Change working directory temporarily to RADAR_inference to resolve relative paths cleanly
    current_cwd = os.getcwd()
    try:
        os.chdir(RADAR_INFERENCE_DIR)
        from RADAR_inference.inference_demo import initialize
        _PAD_FUNC, _MODEL = initialize()
        print("[✓] DAMO RADAR initialized and ready in GPU memory.")
    finally:
        os.chdir(current_cwd)

    return _PAD_FUNC, _MODEL


def run_radar_inference(nii_path: str) -> dict:
    """
    Executes in-memory inference on a single 3D volume, parses the raw 
    probabilities into clean English findings, and purges all scratch files.
    
    Returns:
        dict: {english_finding_name: probability_float}
    """
    pad_func, model = warmup_radar_model()

    from RADAR_inference.inference_demo import evaluate

    with tempfile.TemporaryDirectory() as temp_run_dir:
        input_dir = os.path.join(temp_run_dir, "input")
        save_dir = os.path.join(temp_run_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Place symlink to volume in input_dir
        symlink_name = os.path.basename(nii_path)
        symlink_path = os.path.join(input_dir, symlink_name)
        if not os.path.exists(symlink_path):
            os.symlink(os.path.abspath(nii_path), symlink_path)

        # Execute native inference
        current_cwd = os.getcwd()
        try:
            os.chdir(RADAR_INFERENCE_DIR)
            evaluate(pad_func, model, input_dir, save_dir, save_tag="service")
        finally:
            os.chdir(current_cwd)

        # Parse generated output into clean English findings
        csv_path = os.path.join(save_dir, "RADAR_infer_results_service.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"RADAR failed to generate results for volume: {nii_path}")

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

    # Scratch directory is destroyed upon leaving context
    return predictions