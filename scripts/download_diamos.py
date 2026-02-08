"""Download and prepare DiaMOS Plant dataset from Zenodo.

DiaMOS Plant: 3,505 pear leaf images with severity annotations (0-4)
and 4 disease classes (healthy, spot, curl, slug).

Source: https://zenodo.org/records/5557313

Usage:
    python scripts/download_diamos.py --output_dir data/diamos

The dataset is ~13GB. If automatic download fails, download manually from Zenodo
and extract to the output directory.
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ZENODO_RECORD_ID = "5557313"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"


def download_with_urllib(url: str, dest: Path) -> bool:
    """Download a file with progress reporting."""
    import urllib.request

    logger.info(f"Downloading from {url} ...")
    logger.info(f"Destination: {dest}")

    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=report_progress)
        print()  # newline after progress
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract zip file to output directory."""
    logger.info(f"Extracting {zip_path} to {output_dir} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(output_dir))
    logger.info("Extraction complete.")


def verify_dataset(data_dir: Path) -> bool:
    """Check that the DiaMOS dataset structure looks correct."""
    # Look for image files and annotation CSVs
    image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [f for f in data_dir.rglob("*") if f.suffix in image_exts]
    csvs = list(data_dir.rglob("*.csv"))

    logger.info(f"Found {len(images)} images and {len(csvs)} CSV files in {data_dir}")

    if len(images) == 0:
        logger.warning("No images found. Dataset may not be properly extracted.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Download DiaMOS Plant dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/diamos",
        help="Directory to store the dataset",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download, only verify/prepare an already extracted dataset",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        # Zenodo direct download link for the dataset zip
        # The actual file URLs can be found on the Zenodo record page
        download_url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/Pear.zip?download=1"
        zip_path = output_dir / "Pear.zip"

        if zip_path.exists():
            logger.info(f"Zip already exists at {zip_path}, skipping download.")
        else:
            success = download_with_urllib(download_url, zip_path)
            if not success:
                logger.error(
                    f"Automatic download failed.\n"
                    f"Please download manually from: {ZENODO_URL}\n"
                    f"Save the zip file to: {zip_path}\n"
                    f"Then re-run with: python scripts/download_diamos.py --skip_download"
                )
                sys.exit(1)

        # Extract
        if zip_path.exists():
            extract_zip(zip_path, output_dir)

    # Verify
    if verify_dataset(output_dir):
        logger.info("DiaMOS dataset ready.")
    else:
        logger.warning(
            f"Dataset verification failed. Check {output_dir} contents.\n"
            f"Expected: image files and CSV annotations from DiaMOS Plant."
        )


if __name__ == "__main__":
    main()
