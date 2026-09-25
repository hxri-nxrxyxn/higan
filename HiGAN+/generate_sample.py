import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from lib.utils import yaml2config
from lib.alphabet import strLabelConverter
from networks import get_model
from networks.utils import rescale_images2

def main():
    import sys
    use_cpu = "--cpu" in sys.argv or not torch.cuda.is_available()
    if not use_cpu:
        try:
            # test a small cuda allocation
            _ = torch.zeros(1, device='cuda:0')
            device = torch.device('cuda:0')
        except Exception:
            print("GPU memory unavailable (e.g. ComfyUI in use), falling back to CPU...")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    config_path = "configs/gan_image.yml"
    ckpt_path = "pretrained/deploy_HiGAN+.pth"
    cfg = yaml2config(config_path)
    cfg.device = str(device)

    print("Initializing HiGAN+ model...")
    model = get_model(cfg.model)(cfg, config_path)
    model.load(ckpt_path, device)
    model.set_mode('eval')

    label_converter = strLabelConverter('all')
    org_transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])

    # Load reference style images from data/image_samples
    samples_dir = "./data/image_samples"
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.png')]
    print(f"Found {len(sample_files)} style samples in {samples_dir}: {sample_files}")

    output_dir = "./output/test_generation"
    os.makedirs(output_dir, exist_ok=True)

    test_words = ["Hello", "DeepLearning", "Assignment", "ComputerVision", "HiGANplus", "Synthesized"]

    with torch.no_grad():
        for sample_fn in sample_files[:3]: # test top 3 styles
            style_name = os.path.splitext(sample_fn)[0]
            img_path = os.path.join(samples_dir, sample_fn)
            ref_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = ref_cv.shape[:2]

            # Normalize height to 64
            r = 64.0 / float(h)
            new_w = max(int(w * r), int(64 / 4 * len(style_name)))
            resized_ref = cv2.resize(ref_cv, (new_w, 64), interpolation=cv2.INTER_AREA)
            # Invert: white background -> black background
            inverted_ref = 255 - resized_ref

            ref_tensor = org_transforms(Image.fromarray(inverted_ref, mode='L')).unsqueeze(0).to(device)
            ref_len = torch.IntTensor([new_w]).to(device)
            ref_lb_len = torch.IntTensor([len(style_name)]).to(device)

            # Extract style embedding (32-dim vector)
            enc_style = model.models.E(ref_tensor, ref_len, model.models.B)
            print(f"Extracted style embedding for '{style_name}' (vector norm: {enc_style.norm().item():.3f})")

            # Generate each test word
            generated_word_imgs = []
            for word in test_words:
                fake_lb = torch.LongTensor(label_converter.encode(word)).unsqueeze(0).to(device)
                fake_lb_len = torch.IntTensor([len(word)]).to(device)

                gen_img = model.models.G(enc_style, fake_lb, fake_lb_len)
                gen_img, _ = rescale_images2(gen_img, fake_lb_len * cfg.char_width, fake_lb_len, ref_len, ref_lb_len)

                # Convert to black ink on white background: (1 - img) * 127
                gen_np = (1 - gen_img).squeeze().cpu().numpy() * 127
                gen_np = np.clip(gen_np, 0, 255).astype(np.uint8)
                generated_word_imgs.append(gen_np)

                # Save individual word
                word_out_path = os.path.join(output_dir, f"{style_name}_{word}.png")
                cv2.imwrite(word_out_path, gen_np)

            # Combine words into a single line image with spacing
            gap = np.ones((64, 25), dtype=np.uint8) * 255
            combined = []
            for g_img in generated_word_imgs:
                combined.append(g_img)
                combined.append(gap)
            combined_line = np.hstack(combined[:-1])
            combined_out_path = os.path.join(output_dir, f"line_{style_name}.png")
            cv2.imwrite(combined_out_path, combined_line)
            print(f"Saved generated line: {combined_out_path}")

    print("Generation complete! All outputs saved to:", output_dir)

if __name__ == '__main__':
    main()
