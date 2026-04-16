"""
Step 0a: Download and extract Video-SafetyBench videos.

Run once before anything else:
    python scripts/download_data.py

Downloads video.tar.gz from HuggingFace and extracts to data/videos/.
The video_path field in the dataset (e.g. "video/1_Violent_Crimes/Animal_Abuse/1.mp4")
will resolve to data/videos/1_Violent_Crimes/Animal_Abuse/1.mp4 after extraction.
"""
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

REPO_ID = "BAAI/Video-SafetyBench"
VIDEO_ARCHIVE = "video.tar.gz"
OUTPUT_DIR = Path("data/videos")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading video.tar.gz from HuggingFace...")
    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=VIDEO_ARCHIVE,
        repo_type="dataset",
    )

    print(f"Extracting to {OUTPUT_DIR}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(OUTPUT_DIR)

    print(f"Done. Videos extracted to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
