#!/usr/bin/env python3
"""
Synthesize low-light remote sensing images from well-exposed ones.

Per-image degradation pipeline (applied in this exact order):
    1. Global exposure attenuation
    2. Per-channel spectral attenuation (atmospheric scattering)
    3. Photon shot noise (Poisson)
    4. Readout noise (Gaussian)
    5. Dark current offset
    6. Row-wise stripe noise (probabilistic)
    7. Quantization

Usage:
    - Edit SOURCE_DIR / OUTPUT_DIR below, then run the script.
    - Multiprocessing is used for throughput; already-existing output
      files are skipped, so the run can be safely resumed.
"""

import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
SOURCE_DIR = Path("/root/autodl-tmp/DOTA_high_clean")
OUTPUT_DIR = Path("/root/autodl-tmp/DOTA_low_light_clean2")
SPLITS = ("train", "val")
RANDOM_SEED = 42
NUM_WORKERS = 8  # adjust to the number of CPU cores

# Shot-noise model: pixel intensities are mapped to equivalent electron
# counts before Poisson sampling, then mapped back.
ELECTRON_FULL_WELL = 2000.0


def degrade_image(img_bgr: np.ndarray) -> np.ndarray:
    """Apply the low-light degradation pipeline to one BGR image.

    Input/output are uint8 BGR arrays in [0, 255]; all intermediate
    computation is done in float32 in [0, 1].
    """
    img = img_bgr.astype(np.float32) / 255.0

    # 1. Global exposure attenuation.
    exposure = np.random.uniform(0.15, 0.50)
    img = img * exposure

    # 2. Per-channel spectral attenuation (BGR order, OpenCV convention).
    #    Blue is attenuated most, red least, to mimic atmospheric scattering.
    img[:, :, 0] *= np.random.uniform(0.85, 0.95)
    img[:, :, 1] *= np.random.uniform(0.90, 0.98)
    img[:, :, 2] *= np.random.uniform(0.95, 1.00)

    # 3. Photon shot noise.
    electrons = np.maximum(img * ELECTRON_FULL_WELL, 1e-6)
    img = np.random.poisson(electrons) / ELECTRON_FULL_WELL

    # 4. Readout noise.
    read_noise_sigma = np.random.uniform(0.0005, 0.002)
    img = img + np.random.normal(0, read_noise_sigma, img.shape)

    # 5. Dark current offset.
    dark_current = np.random.uniform(0.0005, 0.0015)
    img = img + dark_current

    # 6. Row-wise stripe noise (applied with probability 0.3).
    if np.random.rand() < 0.3:
        stripe_amp = np.random.uniform(0.0003, 0.001)
        stripe = np.random.normal(0, stripe_amp, img.shape[0])
        img += stripe[:, np.newaxis, np.newaxis]

    # 7. Quantization at a random bit depth in [12, 14].
    levels = 2 ** np.random.randint(12, 15) - 1
    img = np.round(img * levels) / levels

    return np.clip(img * 255, 0, 255).astype(np.uint8)


def process_one_image(img_path: Path, dst_dir: Path) -> str:
    """Degrade one image and write it to dst_dir. Returns a status string."""
    dst_path = dst_dir / img_path.name
    if dst_path.exists():  # resume support: skip finished outputs
        return f"Skipped {img_path.name} (already exists)"

    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return f"Failed to read {img_path.name}"
        dark_bgr = degrade_image(img_bgr)
        cv2.imwrite(str(dst_path), dark_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return f"Processed {img_path.name}"
    except Exception as e:
        return f"Error {img_path.name}: {e}"


def process_split(split: str, src_dir: Path, dst_dir: Path) -> None:
    """Degrade all PNG images of one split, in parallel."""
    src_split = src_dir / split
    dst_split = dst_dir / split
    dst_split.mkdir(parents=True, exist_ok=True)

    images = list(src_split.glob("*.png"))
    if not images:
        print(f"{split}: no images")
        return

    print(f"{split}: {len(images)} images, using {NUM_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_one_image, img, dst_split): img
                   for img in images}
        for i, future in enumerate(as_completed(futures)):
            if i % 50 == 0:
                print(f"  {split} progress: {i}/{len(images)}", flush=True)
    print(f"  {split} done")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {NUM_WORKERS}")
    print()

    for split in SPLITS:
        process_split(split, SOURCE_DIR, OUTPUT_DIR)

    print("\nAll done.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal time: {(time.time() - start) / 60:.2f} minutes")
