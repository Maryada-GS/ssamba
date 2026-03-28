"""Download SSAMBA pretrained weights from Google Drive.

Usage:
    python download_weights.py ssamba_tiny_250

If the requested .pth file already exists in src/model_weights/, the download
is skipped. Otherwise the full weights folder is downloaded via gdown.
"""

import os
import sys


GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX"

# Resolve src/model_weights/ relative to this script's location:
# this file is at src/finetune/speechcommands_v2/download_weights.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "model_weights")
MODEL_WEIGHTS_DIR = os.path.normpath(MODEL_WEIGHTS_DIR)


def download_weights(model_name):
    """Download model_name.pth if not already present.

    Args:
        model_name: e.g. "ssamba_tiny_250"
    """
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is not installed. Run: pip install gdown")
        sys.exit(1)

    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)

    target_path = os.path.join(MODEL_WEIGHTS_DIR, f"{model_name}.pth")

    if os.path.isfile(target_path):
        print(f"[download_weights] {model_name}.pth already exists at {target_path}, skipping.")
        return

    print(f"[download_weights] {model_name}.pth not found.")
    print(f"[download_weights] Downloading full weights folder from Google Drive...")
    print(f"[download_weights] Destination: {MODEL_WEIGHTS_DIR}")

    gdown.download_folder(
        url=GDRIVE_FOLDER_URL,
        output=MODEL_WEIGHTS_DIR,
        quiet=False,
        use_cookies=False,
    )

    if not os.path.isfile(target_path):
        print(f"ERROR: Download completed but {model_name}.pth not found in {MODEL_WEIGHTS_DIR}.")
        print("Please download manually from:")
        print(f"  {GDRIVE_FOLDER_URL}")
        sys.exit(1)

    print(f"[download_weights] {model_name}.pth ready at {target_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <model_name>")
        print("Example: python download_weights.py ssamba_tiny_250")
        sys.exit(1)

    download_weights(sys.argv[1])
