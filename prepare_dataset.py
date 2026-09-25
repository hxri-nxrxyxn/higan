import os
import sys
import argparse
import cv2
import numpy as np

def clean_and_segment_page(image_path: str, output_dir: str = "data/my_handwriting", min_w=30, min_h=20, max_h=250):
    """
    Cleans a smartphone photo of a handwritten page and segments individual words.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Illumination normalization / shadow removal
    # Estimate background illumination with large morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    # 2. Binarization (Otsu thresholding)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Morphological dilation to connect letters within words horizontally
    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, word_kernel, iterations=1)

    # 4. Find word contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and min_h <= h <= max_h:
            boxes.append((x, y, w, h))

    # Sort boxes top-to-bottom, left-to-right (line reading order)
    # Group boxes with similar y coordinates into lines
    boxes = sorted(boxes, key=lambda b: (b[1] // 40, b[0]))

    print(f"Detected {len(boxes)} word regions in {image_path}")

    saved_files = []
    for idx, (x, y, w, h) in enumerate(boxes):
        # Add slight padding
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        # Extract normalized grayscale patch (pure white paper, dark ink)
        crop_clean = norm[y1:y2, x1:x2]

        # Enhance contrast
        crop_clean = cv2.normalize(crop_clean, None, 0, 255, cv2.NORM_MINMAX)

        out_fn = f"sample_{idx+1:03d}.png"
        out_fp = os.path.join(output_dir, out_fn)
        cv2.imwrite(out_fp, crop_clean)
        saved_files.append(out_fp)

    print(f"Extracted {len(saved_files)} word crops to '{output_dir}/'")
    print("Next: Rename each image to the word it shows (e.g. rename sample_001.png to 'because.png')")
    return saved_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop and clean words from handwriting page photo")
    parser.add_argument("--image", type=str, required=True, help="Path to photo of your handwriting")
    parser.add_argument("--output_dir", type=str, default="data/my_handwriting", help="Output directory")
    args = parser.parse_args()

    clean_and_segment_page(args.image, args.output_dir)
