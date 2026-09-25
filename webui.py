import os
import sys
import time
import uuid
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from torchvision.transforms import Compose, Normalize, ToTensor

# Ensure HiGAN+ is in path
repo_root = os.path.dirname(os.path.abspath(__file__))
higan_dir = os.path.join(repo_root, "HiGAN+")
if higan_dir not in sys.path:
    sys.path.insert(0, higan_dir)

from lib.utils import yaml2config
from lib.alphabet import strLabelConverter
from networks import get_model
from networks.utils import rescale_images2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(repo_root, "uploads")
app.config['OUTPUT_FOLDER'] = os.path.join(repo_root, "output")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global model state
MODEL = None
CFG = None
DEVICE = None
LABEL_CONVERTER = None
ORG_TRANSFORMS = None

def init_model(device_str="auto"):
    global MODEL, CFG, DEVICE, LABEL_CONVERTER, ORG_TRANSFORMS
    if MODEL is not None:
        return

    if device_str == "auto":
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(device_str)

    prev_cwd = os.getcwd()
    os.chdir(higan_dir)
    config_path = "configs/gan_image.yml"
    ckpt_path = "pretrained/deploy_HiGAN+.pth"

    CFG = yaml2config(config_path)
    CFG.device = str(DEVICE)

    print(f"Loading HiGAN+ model onto {DEVICE}...")
    MODEL = get_model(CFG.model)(CFG, config_path)
    MODEL.load(ckpt_path, DEVICE)
    MODEL.set_mode('eval')
    os.chdir(prev_cwd)

    LABEL_CONVERTER = strLabelConverter('all')
    ORG_TRANSFORMS = Compose([ToTensor(), Normalize([0.5], [0.5])])
    print("HiGAN+ model initialized and ready!")

def render_handwriting(text: str, style_image_path: str, output_path: str):
    ref_cv = cv2.imread(style_image_path, cv2.IMREAD_GRAYSCALE)
    if ref_cv is None:
        raise ValueError(f"Could not load style image from {style_image_path}")

    h, w = ref_cv.shape[:2]
    style_name = os.path.splitext(os.path.basename(style_image_path))[0]
    r = 64.0 / float(h)
    new_w = max(int(w * r), int(64 / 4 * len(style_name)))
    resized_ref = cv2.resize(ref_cv, (new_w, 64), interpolation=cv2.INTER_AREA)

    if np.mean(resized_ref) > 127:
        inverted_ref = 255 - resized_ref
    else:
        inverted_ref = resized_ref

    ref_tensor = ORG_TRANSFORMS(Image.fromarray(inverted_ref, mode='L')).unsqueeze(0).to(DEVICE)
    ref_len = torch.IntTensor([new_w]).to(DEVICE)
    ref_lb_len = torch.IntTensor([max(len(style_name), 1)]).to(DEVICE)

    with torch.no_grad():
        enc_style = MODEL.models.E(ref_tensor, ref_len, MODEL.models.B)
        lines = text.split("\n")
        rendered_lines = []

        for line in lines:
            words = line.strip().split()
            if not words:
                blank = np.ones((64, 100), dtype=np.uint8) * 255
                rendered_lines.append(blank)
                continue

            generated_words = []
            for word in words:
                cleaned_word = "".join([c for c in word if c in LABEL_CONVERTER.dict])
                if not cleaned_word:
                    continue

                fake_lb = torch.LongTensor(LABEL_CONVERTER.encode(cleaned_word)).unsqueeze(0).to(DEVICE)
                fake_lb_len = torch.IntTensor([len(cleaned_word)]).to(DEVICE)

                gen_img = MODEL.models.G(enc_style, fake_lb, fake_lb_len)
                gen_img, _ = rescale_images2(gen_img, fake_lb_len * CFG.char_width, fake_lb_len, ref_len, ref_lb_len)

                gen_np = (1 - gen_img).squeeze().cpu().numpy() * 127
                gen_np = np.clip(gen_np, 0, 255).astype(np.uint8)
                generated_words.append(gen_np)

            if not generated_words:
                continue

            space_gap = np.ones((64, 25), dtype=np.uint8) * 255
            line_pieces = []
            for gw in generated_words:
                line_pieces.append(gw)
                line_pieces.append(space_gap)
            combined_line = np.hstack(line_pieces[:-1])
            rendered_lines.append(combined_line)

        if not rendered_lines:
            raise ValueError("No readable words found in input text.")

        max_line_w = max(l.shape[1] for l in rendered_lines)
        line_spacing = 20
        padded_lines = []
        for l in rendered_lines:
            pad_w = max_line_w - l.shape[1]
            if pad_w > 0:
                padding = np.ones((64, pad_w), dtype=np.uint8) * 255
                l = np.hstack([l, padding])
            padded_lines.append(l)

        vgap = np.ones((line_spacing, max_line_w), dtype=np.uint8) * 255
        doc_pieces = []
        for pl in padded_lines:
            doc_pieces.append(pl)
            doc_pieces.append(vgap)
        final_doc = np.vstack(doc_pieces[:-1])
        final_doc = cv2.copyMakeBorder(final_doc, 30, 30, 40, 40, cv2.BORDER_CONSTANT, value=255)

        cv2.imwrite(output_path, final_doc)
        return output_path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HiGAN+ Handwriting Synthesizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    h1 { font-size: 1.6rem; font-weight: 700; color: #60a5fa; }
    .badge {
      font-size: 0.8rem;
      background: #1e3a8a;
      color: #93c5fd;
      padding: 4px 10px;
      border-radius: 9999px;
      font-weight: 600;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 32px;
    }
    @media (max-width: 768px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    h2 { font-size: 1.15rem; font-weight: 600; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    label { display: block; font-size: 0.9rem; font-weight: 500; margin-bottom: 6px; color: var(--text-muted); }
    textarea, select, input[type="text"] {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 0.95rem;
      font-family: inherit;
      resize: vertical;
      margin-bottom: 16px;
    }
    textarea:focus, select:focus, input:focus {
      outline: none;
      border-color: var(--accent);
    }
    .style-picker {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
      max-height: 200px;
      overflow-y: auto;
      padding: 4px;
    }
    .style-card {
      border: 2px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
      background: #0f172a;
    }
    .style-card:hover { border-color: var(--accent); }
    .style-card.active { border-color: var(--accent); background: #1e3a8a33; }
    .style-card img { max-width: 100%; height: 32px; object-fit: contain; filter: invert(1); margin-bottom: 4px; }
    .style-card .name { font-size: 0.75rem; color: var(--text-muted); word-break: break-all; }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      padding: 12px;
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .preview-box {
      min-height: 220px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: #0f172a;
      border: 1px dashed var(--border);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }
    .preview-box img {
      max-width: 100%;
      height: auto;
      background: white;
      border-radius: 4px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .preview-actions {
      margin-top: 12px;
      display: flex;
      gap: 12px;
      width: 100%;
    }
    .btn-secondary {
      background: #334155;
      padding: 8px 16px;
      border-radius: 6px;
      color: white;
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 500;
      text-align: center;
    }
    .btn-secondary:hover { background: #475569; }
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }
    .gallery-item {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .gallery-item img {
      width: 100%;
      height: 100px;
      object-fit: contain;
      background: white;
      border-radius: 4px;
      margin-bottom: 8px;
    }
    .gallery-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .file-input-wrapper {
      margin-bottom: 16px;
      padding: 10px;
      border: 1px dashed var(--border);
      border-radius: 8px;
      background: #0f172a;
      text-align: center;
    }
    .file-input-wrapper input { display: none; }
    .file-input-label {
      cursor: pointer;
      color: var(--accent);
      font-size: 0.85rem;
      font-weight: 500;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>HiGAN+ Handwriting Synthesizer</h1>
        <p style="color: var(--text-muted); font-size: 0.85rem;">Synthesize realistic handwritten assignments from text</p>
      </div>
      <div>
        <span class="badge">Running on {{ device }}</span>
      </div>
    </header>

    <div class="grid">
      <!-- Input Controls -->
      <div class="card">
        <h2>1. Synthesis Prompt & Style</h2>
        <form id="synthForm">
          <label for="textInput">Text to Write (supports multi-line):</label>
          <textarea id="textInput" rows="5" placeholder="Enter assignment text here...&#10;e.g.&#10;Assignment 1: Neural Networks&#10;Question 1: What is backpropagation?">Assignment 1: Artificial Intelligence&#10;Submitted by: Hari Narayan&#10;Question 1: Explain generative adversarial networks.</textarea>

          <label>Select Reference Handwriting Style:</label>
          <div class="style-picker" id="stylePicker">
            {% for sample in samples %}
            <div class="style-card {% if sample.name == 'preparation' %}active{% endif %}" data-style="{{ sample.name }}">
              <img src="/data/image_samples/{{ sample.filename }}" alt="{{ sample.name }}">
              <div class="name">{{ sample.name }}</div>
            </div>
            {% endfor %}
          </div>
          <input type="hidden" id="selectedStyle" value="preparation">

          <label>Or Upload Your Own Handwriting Photo:</label>
          <div class="file-input-wrapper">
            <label class="file-input-label" for="customImageInput">📁 Click to choose an image of your handwriting</label>
            <input type="file" id="customImageInput" accept="image/*">
            <div id="fileNameDisplay" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;"></div>
          </div>

          <button type="submit" class="btn" id="generateBtn">✍️ Generate Handwriting</button>
        </form>
      </div>

      <!-- Live Preview -->
      <div class="card">
        <h2>2. Live Preview</h2>
        <div class="preview-box" id="previewBox">
          <img id="previewImage" src="/output/assignment_demo.png" alt="Generated Handwriting Preview" onerror="this.style.display='none'">
          <p id="placeholderText" style="display: none; color: var(--text-muted);">Click "Generate Handwriting" to render</p>
        </div>
        <div class="preview-actions" id="previewActions">
          <a id="downloadBtn" href="/output/assignment_demo.png" download="handwritten_assignment.png" class="btn-secondary" style="flex: 1;">⬇️ Download Full Image</a>
          <a id="openTabBtn" href="/output/assignment_demo.png" target="_blank" class="btn-secondary">🔗 Open in New Tab</a>
        </div>
      </div>
    </div>

    <!-- Gallery of Generated Files -->
    <div class="card">
      <h2>3. Previously Generated Assignments & Demos</h2>
      <div class="gallery-grid" id="galleryGrid">
        {% for item in gallery %}
        <div class="gallery-item">
          <a href="/output/{{ item.filename }}" target="_blank">
            <img src="/output/{{ item.filename }}" alt="{{ item.filename }}">
          </a>
          <div class="gallery-meta">
            <span>{{ item.filename }}</span>
            <a href="/output/{{ item.filename }}" download style="color: var(--accent); text-decoration: none;">⬇️ Save</a>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>

  <script>
    // Style selection handler
    document.querySelectorAll('.style-card').forEach(card => {
      card.addEventListener('click', () => {
        document.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        document.getElementById('selectedStyle').value = card.dataset.style;
        // Clear file input if built-in style chosen
        document.getElementById('customImageInput').value = '';
        document.getElementById('fileNameDisplay').textContent = '';
      });
    });

    // Custom file input handler
    document.getElementById('customImageInput').addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        document.getElementById('fileNameDisplay').textContent = 'Selected: ' + e.target.files[0].name;
        document.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
      }
    });

    // Form submit
    document.getElementById('synthForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('generateBtn');
      const text = document.getElementById('textInput').value.trim();
      const style = document.getElementById('selectedStyle').value;
      const fileInput = document.getElementById('customImageInput');

      if (!text) {
        alert('Please enter some text to write.');
        return;
      }

      btn.disabled = true;
      btn.textContent = '⏳ Rendering Handwriting...';

      const formData = new FormData();
      formData.append('text', text);
      formData.append('style', style);
      if (fileInput.files.length > 0) {
        formData.append('custom_file', fileInput.files[0]);
      }

      try {
        const resp = await fetch('/api/generate', {
          method: 'POST',
          body: formData
        });
        const data = await resp.json();
        if (data.success) {
          const previewImg = document.getElementById('previewImage');
          const cacheBuster = '?t=' + new Date().getTime();
          previewImg.src = data.url + cacheBuster;
          previewImg.style.display = 'block';
          document.getElementById('placeholderText').style.display = 'none';
          document.getElementById('downloadBtn').href = data.url;
          document.getElementById('openTabBtn').href = data.url;

          // Prepend to gallery
          const gallery = document.getElementById('galleryGrid');
          const newItem = document.createElement('div');
          newItem.className = 'gallery-item';
          newItem.innerHTML = `
            <a href="${data.url}" target="_blank">
              <img src="${data.url + cacheBuster}" alt="${data.filename}">
            </a>
            <div class="gallery-meta">
              <span>${data.filename}</span>
              <a href="${data.url}" download style="color: var(--accent); text-decoration: none;">⬇️ Save</a>
            </div>
          `;
          gallery.insertBefore(newItem, gallery.firstChild);
        } else {
          alert('Error: ' + data.error);
        }
      } catch (err) {
        alert('Request failed: ' + err);
      } finally {
        btn.disabled = false;
        btn.textContent = '✍️ Generate Handwriting';
      }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    samples_dir = os.path.join(higan_dir, "data/image_samples")
    sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.png')])
    samples = [{"name": os.path.splitext(f)[0], "filename": f} for f in sample_files]

    # Load gallery
    out_files = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Sort by mtime descending
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['OUTPUT_FOLDER'], x)), reverse=True)
        out_files = [{"filename": f} for f in files]

    return render_template_string(HTML_TEMPLATE, samples=samples, gallery=out_files, device=str(DEVICE))

@app.route('/data/image_samples/<filename>')
def serve_sample(filename):
    return send_from_directory(os.path.join(higan_dir, "data/image_samples"), filename)

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    text = request.form.get('text', '').strip()
    style = request.form.get('style', 'preparation').strip()

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    # Check if custom image uploaded
    if 'custom_file' in request.files and request.files['custom_file'].filename:
        uploaded_file = request.files['custom_file']
        ext = os.path.splitext(uploaded_file.filename)[1] or '.png'
        custom_fn = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], custom_fn)
        uploaded_file.save(style_path)
    else:
        # built in sample
        sample_candidate = os.path.join(higan_dir, "data/image_samples", style + ".png")
        if os.path.exists(sample_candidate):
            style_path = sample_candidate
        else:
            style_path = os.path.join(higan_dir, "data/image_samples", "preparation.png")

    out_filename = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)

    try:
        render_handwriting(text, style_path, out_path)
        return jsonify({
            "success": True,
            "filename": out_filename,
            "url": f"/output/{out_filename}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    init_model(args.device)
    print(f"\n🚀 HiGAN+ Web UI starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
