# # # import os
# # # import sys

# # # BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # # RADAR_INFERENCE_DIR = os.path.join(BASE_DIR, "RADAR_inference")

# # # if RADAR_INFERENCE_DIR not in sys.path:
# # #     sys.path.insert(0, RADAR_INFERENCE_DIR)
# # # if BASE_DIR not in sys.path:
# # #     sys.path.insert(0, BASE_DIR)

# # # from RADAR_inference.inference_demo import run_model

# # # def run_radar_inference(nii_path: str) -> dict:
# # #     """
# # #     Runs RADAR forward pass directly on the NIfTI volume.
# # #     Returns: dict of {finding_name: probability}
# # #     """
# # #     results_df = run_model(nii_path)
# # #     return dict(zip(results_df["finding"], results_df["probability"].astype(float)))


# # import os
# # import sys
# # import tempfile
# # import pandas as pd

# # BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
# # RADAR_INFERENCE_DIR = os.path.join(BASE_DIR, "RADAR_inference")

# # # Ensure required search paths and environment variables
# # if RADAR_INFERENCE_DIR not in sys.path:
# #     sys.path.insert(0, RADAR_INFERENCE_DIR)
# # if BASE_DIR not in sys.path:
# #     sys.path.insert(0, BASE_DIR)

# # os.environ["MODEL_ROOT"] = CKPT_DIR
# # os.environ["CONFIGS_ROOT"] = CKPT_DIR
# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# # def run_radar_inference(nii_path: str) -> dict:
# #     """
# #     Executes single-volume RADAR inference using infer_single.py,
# #     reads output probabilities into memory, and cleans up the temporary CSV.
# #     """
# #     infer_script = os.path.join(RADAR_INFERENCE_DIR, "infer_single.py")
# #     if not os.path.exists(infer_script):
# #         raise FileNotFoundError(f"Inference script missing: {infer_script}")

# #     with tempfile.TemporaryDirectory() as tmp_dir:
# #         output_csv = os.path.join(tmp_dir, "radar_preds.csv")

# #         # Execute infer_single.py
# #         cmd = (
# #             f"python '{infer_script}' "
# #             f"--img_path '{nii_path}' "
# #             f"--output_csv '{output_csv}'"
# #         )
# #         ret = os.system(cmd)
# #         if ret != 0 or not os.path.exists(output_csv):
# #             raise RuntimeError(f"RADAR inference process crashed or failed with exit code: {ret}")

# #         # Parse CSV output into a clean dictionary
# #         df = pd.read_csv(output_csv)
# #         if df.empty:
# #             return {}

# #         predictions = {}
# #         # Case A: Standard tall format (finding, probability)
# #         if "finding" in df.columns and "probability" in df.columns:
# #             for _, row in df.iterrows():
# #                 predictions[str(row["finding"]).strip()] = float(row["probability"])
# #         # Case B: Wide tabular format (each finding is a column name)
# #         else:
# #             row = df.iloc[0].to_dict()
# #             for col, val in row.items():
# #                 if col not in ["file_name", "index", "id"]:
# #                     clean_finding = col.split("_(")[0].strip()
# #                     try:
# #                         predictions[clean_finding] = float(val)
# #                     except (ValueError, TypeError):
# #                         continue

# #         return predictions




# import os
# import sys
# import torch

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
# RADAR_INFERENCE_DIR = os.path.join(BASE_DIR, "RADAR_inference")

# if RADAR_INFERENCE_DIR not in sys.path:
#     sys.path.insert(0, RADAR_INFERENCE_DIR)
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)

# os.environ["MODEL_ROOT"] = CKPT_DIR
# os.environ["CONFIGS_ROOT"] = CKPT_DIR
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Checkpoint paths
# CHECKPOINT_PATH = os.path.join(CKPT_DIR, "checkpoint_radar_pretrain.pth")
# TEXT_EMBED_PATH = os.path.join(CKPT_DIR, "infer_text_embedding_radar.pt")

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Global model state holders for service warm-up
# _MODEL = None
# _TEXT_EMBEDDINGS = None
# _FINDING_LABELS = None


# def load_radar_model():
#     """Loads model weights and precomputed text embeddings into memory once."""
#     global _MODEL, _TEXT_EMBEDDINGS, _FINDING_LABELS
#     if _MODEL is not None:
#         return _MODEL, _TEXT_EMBEDDINGS, _FINDING_LABELS

#     print("[*] Loading RADAR checkpoint into memory...")
#     # Load precomputed text embeddings dictionary: {finding_name: tensor}
#     if os.path.exists(TEXT_EMBED_PATH):
#         embed_data = torch.load(TEXT_EMBED_PATH, map_location=DEVICE)
#         if isinstance(embed_data, dict):
#             _FINDING_LABELS = list(embed_data.keys())
#             _TEXT_EMBEDDINGS = torch.stack([embed_data[k] for k in _FINDING_LABELS]).to(DEVICE)
#         else:
#             _TEXT_EMBEDDINGS = embed_data.to(DEVICE)

#     # Load weights
#     ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     print("[✓] RADAR weights loaded into GPU memory.")
#     return _MODEL, _TEXT_EMBEDDINGS, _FINDING_LABELS


# def run_radar_inference(nii_path: str) -> dict:
#     """
#     Executes in-memory inference directly on the 3D volume.
#     Returns:
#         dict: {finding_name: probability_float}
#     """
#     # Import RADAR's native inference logic directly as a module
#     from RADAR_inference import inference_demo

#     # Ensure model is warm
#     load_radar_model()

#     # Call the native prediction function directly in Python memory
#     # (Matches inference_demo's internal forward pass)
#     with torch.no_grad():
#         if hasattr(inference_demo, "predict_single_case"):
#             predictions = inference_demo.predict_single_case(nii_path)
#         elif hasattr(inference_demo, "main_single"):
#             predictions = inference_demo.main_single(nii_path)
#         elif hasattr(inference_demo, "infer"):
#             predictions = inference_demo.infer(nii_path)
#         else:
#             # Fallback: call the primary execution function defined inside inference_demo
#             predictions = inference_demo.run_inference(nii_path)

#     # Return pure {finding: score} dictionary
#     if isinstance(predictions, dict):
#         return {k: float(v) for k, v in predictions.items()}
    
#     return {}




import os
import sys
import shutil
import tempfile
import pandas as pd
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