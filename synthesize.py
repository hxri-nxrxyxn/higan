import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

# Ensure HiGAN+ is in Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
higan_dir = os.path.join(repo_root, "HiGAN+")
if higan_dir not in sys.path:
    sys.path.insert(0, higan_dir)

from lib.utils import yaml2config
from lib.alphabet import strLabelConverter
from networks import get_model
from networks.utils import rescale_images2

def synthesize(text: str, style_image_path: str, output_path: str = "output/demo.png", device_str: str = "auto"):
    # Ensure relative paths in configs resolve correctly
    prev_cwd = os.getcwd()
    abs_output_path = os.path.abspath(output_path)
    abs_style_path = os.path.abspath(style_image_path) if os.path.exists(style_image_path) else style_image_path
    if device_str == "auto":
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    os.chdir(higan_dir)
    config_path = "configs/gan_image.yml"
    ckpt_path = "pretrained/deploy_HiGAN+.pth"

    cfg = yaml2config(config_path)
    cfg.device = str(device)

    print(f"Loading HiGAN+ model on {device}...")
    model = get_model(cfg.model)(cfg, config_path)
    model.load(ckpt_path, device)
    model.set_mode('eval')

    label_converter = strLabelConverter('all')
    org_transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])

    # Check style image
    if not os.path.exists(abs_style_path):
        # Check if it's a name in data/image_samples
        sample_candidate = os.path.join(higan_dir, "data/image_samples", style_image_path)
        if not sample_candidate.endswith('.png'):
            sample_candidate += '.png'
        if os.path.exists(sample_candidate):
            style_image_path = sample_candidate
        else:
            raise FileNotFoundError(f"Style image not found at '{style_image_path}'")
    else:
        style_image_path = abs_style_path

    print(f"Reading style reference from: {style_image_path}")
    ref_cv = cv2.imread(style_image_path, cv2.IMREAD_GRAYSCALE)
    if ref_cv is None:
        raise ValueError(f"Could not read image from {style_image_path}")

    h, w = ref_cv.shape[:2]
    style_name = os.path.splitext(os.path.basename(style_image_path))[0]
    r = 64.0 / float(h)
    new_w = max(int(w * r), int(64 / 4 * len(style_name)))
    resized_ref = cv2.resize(ref_cv, (new_w, 64), interpolation=cv2.INTER_AREA)

    # Invert image if background is white (HiGAN expects white text on dark background internally)
    # Check if background is mostly light
    if np.mean(resized_ref) > 127:
        inverted_ref = 255 - resized_ref
    else:
        inverted_ref = resized_ref

    ref_tensor = org_transforms(Image.fromarray(inverted_ref, mode='L')).unsqueeze(0).to(device)
    ref_len = torch.IntTensor([new_w]).to(device)
    ref_lb_len = torch.IntTensor([max(len(style_name), 1)]).to(device)

    with torch.no_grad():
        enc_style = model.models.E(ref_tensor, ref_len, model.models.B)
        print(f"Style vector extracted successfully.")

        # Split text into lines and words
        lines = text.split("\n")
        rendered_lines = []

        for line_idx, line in enumerate(lines):
            words = line.strip().split()
            if not words:
                # blank line
                blank = np.ones((64, 100), dtype=np.uint8) * 255
                rendered_lines.append(blank)
                continue

            generated_words = []
            for word in words:
                # Filter out characters not in alphabet
                cleaned_word = "".join([c for c in word if c in label_converter.dict])
                if not cleaned_word:
                    continue

                fake_lb = torch.LongTensor(label_converter.encode(cleaned_word)).unsqueeze(0).to(device)
                fake_lb_len = torch.IntTensor([len(cleaned_word)]).to(device)

                gen_img = model.models.G(enc_style, fake_lb, fake_lb_len)
                gen_img, _ = rescale_images2(gen_img, fake_lb_len * cfg.char_width, fake_lb_len, ref_len, ref_lb_len)

                # Convert to black ink on white background
                gen_np = (1 - gen_img).squeeze().cpu().numpy() * 127
                gen_np = np.clip(gen_np, 0, 255).astype(np.uint8)
                generated_words.append(gen_np)

            if not generated_words:
                continue

            # Join words horizontally with natural word spacing (~20-30px)
            space_gap = np.ones((64, 25), dtype=np.uint8) * 255
            line_pieces = []
            for gw in generated_words:
                line_pieces.append(gw)
                line_pieces.append(space_gap)
            combined_line = np.hstack(line_pieces[:-1])
            rendered_lines.append(combined_line)

        # Pad lines to max width and stack vertically
        if rendered_lines:
            max_line_w = max(l.shape[1] for l in rendered_lines)
            line_spacing = 20 # vertical space between lines
            padded_lines = []
            for l in rendered_lines:
                pad_w = max_line_w - l.shape[1]
                if pad_w > 0:
                    padding = np.ones((64, pad_w), dtype=np.uint8) * 255
                    l = np.hstack([l, padding])
                padded_lines.append(l)

            # Stack with line spacing
            vgap = np.ones((line_spacing, max_line_w), dtype=np.uint8) * 255
            doc_pieces = []
            for pl in padded_lines:
                doc_pieces.append(pl)
                doc_pieces.append(vgap)
            final_doc = np.vstack(doc_pieces[:-1])

            # Add outer border/margin
            final_doc = cv2.copyMakeBorder(final_doc, 30, 30, 40, 40, cv2.BORDER_CONSTANT, value=255)

            os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
            cv2.imwrite(abs_output_path, final_doc)
            os.chdir(prev_cwd)
            print(f"Generated handwriting saved to: {abs_output_path}")
            return abs_output_path
        else:
            os.chdir(prev_cwd)
            print("No words could be rendered.")
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HiGAN+ Handwriting Synthesis Demo")
    parser.add_argument("--text", type=str, default="This is a handwriting synthesis demo with HiGAN+", help="Text to synthesize")
    parser.add_argument("--style", type=str, default="preparation", help="Style name (from data/image_samples) or path to handwriting image")
    parser.add_argument("--output", type=str, default="output/demo.png", help="Path to save generated image")
    parser.add_argument("--device", type=str, default="auto", help="'cuda:0', 'cpu', or 'auto'")

    args = parser.parse_args()
    synthesize(args.text, args.style, args.output, args.device)
