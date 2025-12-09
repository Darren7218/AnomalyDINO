import cv2
import os
import numpy as np
import random
from typing import Tuple
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_IMAGE = "data/PCB_DATASET/PCB_USED/01.JPG"
OUTPUT_TRAIN_GOOD = "data/pcb_anomaly_detection/PCB/train/good"
OUTPUT_TEST_GOOD = "data/pcb_anomaly_detection/PCB/test/good"

NUM_TRAIN_SAMPLES = 20  # Number of augmented training images
NUM_TEST_GOOD_SAMPLES = 10  # Number of test good samples (for evaluation)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Create output directories
os.makedirs(OUTPUT_TRAIN_GOOD, exist_ok=True)
os.makedirs(OUTPUT_TEST_GOOD, exist_ok=True)

# =============================================================================
# SAFE AUGMENTATION FUNCTIONS (Preserve Image Identity) means doesnt change the colour and cropping
# =============================================================================

def add_minimal_noise(img: np.ndarray, std: float = 5.0) -> np.ndarray:
    """Add very slight Gaussian noise - imperceptible but adds variation."""
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def adjust_brightness_slightly(img: np.ndarray, delta_range: int = 15) -> np.ndarray:
    """Very subtle brightness adjustment."""
    delta = random.randint(-delta_range, delta_range)
    adjusted = img.astype(np.float32) + delta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast_slightly(img: np.ndarray, alpha_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Very subtle contrast adjustment."""
    alpha = random.uniform(*alpha_range)
    mean = img.mean()
    adjusted = mean + alpha * (img.astype(np.float32) - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def tiny_rotation(img: np.ndarray, angle_range: float = 3.0) -> np.ndarray:
    """Rotate by a very small angle (±3 degrees max)."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-angle_range, angle_range)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), 
                         borderMode=cv2.BORDER_REFLECT_101,
                         flags=cv2.INTER_LINEAR)


def tiny_translation(img: np.ndarray, pixel_range: int = 10) -> np.ndarray:
    """Translate by a few pixels."""
    h, w = img.shape[:2]
    tx = random.randint(-pixel_range, pixel_range)
    ty = random.randint(-pixel_range, pixel_range)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), 
                         borderMode=cv2.BORDER_REFLECT_101,
                         flags=cv2.INTER_LINEAR)


def slight_blur(img: np.ndarray) -> np.ndarray:
    """Apply minimal Gaussian blur."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def gamma_correction(img: np.ndarray, gamma_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Apply slight gamma correction (lighting variation)."""
    gamma = random.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img, table)


# =============================================================================
# AUGMENTATION STRATEGIES
# =============================================================================

def generate_training_augmentation(img: np.ndarray, aug_id: int) -> np.ndarray:
    """
    Generate training augmentation with controlled variation.
    Strategy: Apply 2-3 weak augmentations per image.
    """
    aug_img = img.copy()
    
    # Set different augmentation combinations based on ID
    # This ensures diversity while keeping changes minimal
    
    if aug_id % 4 == 0:
        # Combination 1: Spatial + Noise
        aug_img = tiny_rotation(aug_img, angle_range=2.0)
        aug_img = add_minimal_noise(aug_img, std=4.0)
        
    elif aug_id % 4 == 1:
        # Combination 2: Translation + Brightness
        aug_img = tiny_translation(aug_img, pixel_range=8)
        aug_img = adjust_brightness_slightly(aug_img, delta_range=12)
        
    elif aug_id % 4 == 2:
        # Combination 3: Contrast + Gamma
        aug_img = adjust_contrast_slightly(aug_img, alpha_range=(0.92, 1.08))
        aug_img = gamma_correction(aug_img, gamma_range=(0.95, 1.05))
        
    else:
        # Combination 4: Light blur + Noise
        aug_img = slight_blur(aug_img)
        aug_img = add_minimal_noise(aug_img, std=3.0)
    
    # Occasionally add a second weak augmentation
    if random.random() < 0.3:
        aug_img = add_minimal_noise(aug_img, std=3.0)
    
    return aug_img


def generate_test_good_augmentation(img: np.ndarray, aug_id: int) -> np.ndarray:
    """
    Generate test good augmentation - VERY SIMILAR to training augmentation.
    These should be from the same distribution as training samples.
    """
    aug_img = img.copy()
    
    # Use same augmentation strategy as training
    if aug_id % 4 == 0:
        aug_img = tiny_rotation(aug_img, angle_range=2.0)
        aug_img = add_minimal_noise(aug_img, std=4.0)
        
    elif aug_id % 4 == 1:
        aug_img = tiny_translation(aug_img, pixel_range=8)
        aug_img = adjust_brightness_slightly(aug_img, delta_range=12)
        
    elif aug_id % 4 == 2:
        aug_img = adjust_contrast_slightly(aug_img, alpha_range=(0.92, 1.08))
        aug_img = gamma_correction(aug_img, gamma_range=(0.95, 1.05))
        
    else:
        aug_img = slight_blur(aug_img)
        aug_img = add_minimal_noise(aug_img, std=3.0)
    
    if random.random() < 0.3:
        aug_img = add_minimal_noise(aug_img, std=3.0)
    
    return aug_img


# =============================================================================
# MAIN GENERATION
# =============================================================================

def main():
    # Load original image
    print(f"Loading image from: {INPUT_IMAGE}")
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{INPUT_IMAGE}'")
    
    print(f"Original image shape: {image.shape}")
    
    # -------------------------------------------------------------------------
    # 1. Generate TRAINING GOOD images
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Generating {NUM_TRAIN_SAMPLES} TRAINING images...")
    print(f"{'='*70}")
    
    # Save original as good_000
    orig_train_path = os.path.join(OUTPUT_TRAIN_GOOD, "good_000.png")
    cv2.imwrite(orig_train_path, image)
    print(f"[TRAIN] Saved original: good_000.png")
    
    # Generate augmented versions
    for i in range(1, NUM_TRAIN_SAMPLES):
        aug_img = generate_training_augmentation(image, i)
        out_path = os.path.join(OUTPUT_TRAIN_GOOD, f"good_{i:03d}.png")
        cv2.imwrite(out_path, aug_img)
        
        if i % 5 == 0:
            print(f"[TRAIN] Generated {i}/{NUM_TRAIN_SAMPLES-1} augmented images...")
    
    print(f"✓ Training dataset complete: {NUM_TRAIN_SAMPLES} images")
    print(f"  Location: {OUTPUT_TRAIN_GOOD}")
    
    # -------------------------------------------------------------------------
    # 2. Generate TEST GOOD images (same distribution as training)
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Generating {NUM_TEST_GOOD_SAMPLES} TEST GOOD images...")
    print(f"{'='*70}")
    
    # IMPORTANT: Test good images should come from SAME distribution as training
    # So we use the same augmentation strategy
    
    # Save original as good_000
    orig_test_path = os.path.join(OUTPUT_TEST_GOOD, "good_000.png")
    cv2.imwrite(orig_test_path, image)
    print(f"[TEST GOOD] Saved original: good_000.png")
    
    # Generate augmented test good samples
    for i in range(1, NUM_TEST_GOOD_SAMPLES):
        aug_img = generate_test_good_augmentation(image, i)
        out_path = os.path.join(OUTPUT_TEST_GOOD, f"good_{i:03d}.png")
        cv2.imwrite(out_path, aug_img)
        
        if i % 5 == 0:
            print(f"[TEST GOOD] Generated {i}/{NUM_TEST_GOOD_SAMPLES-1} augmented images...")
    
    print(f"✓ Test good dataset complete: {NUM_TEST_GOOD_SAMPLES} images")
    print(f"  Location: {OUTPUT_TEST_GOOD}")
    
    # -------------------------------------------------------------------------
    # 3. Summary and recommendations
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DATASET GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"  Training good images:  {NUM_TRAIN_SAMPLES} (in {OUTPUT_TRAIN_GOOD})")
    print(f"  Test good images:      {NUM_TEST_GOOD_SAMPLES} (in {OUTPUT_TEST_GOOD})")
    print(f"\n⚠️  IMPORTANT: DO NOT augment test/bad images!")
    print(f"  Your test bad images should remain as-is (same angle/position/brightness")
    print(f"  as original, but with defects). This ensures they're 'close' to training")
    print(f"  images in feature space, and defects become the discriminating factor.")
    
    print(f"\n🚀 Recommended command:")
    print(f"  python run_anomalydino.py --dataset PCB \\")
    print(f"    --data_root data/pcb_anomaly_detection \\")
    print(f"    --preprocess force_no_mask_no_rotation \\")
    print(f"    --k_neighbors 3 --shots -1 --resolution 448")
    
    print(f"\n💡 Expected results:")
    print(f"  - Good samples: LOW scores (< 0.02)")
    print(f"  - Anomaly samples: HIGH scores (> 0.05)")
    print(f"  - AUROC: > 0.95")


if __name__ == "__main__":
    main()