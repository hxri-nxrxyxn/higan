import os
import sys
import time
import json
import uuid
import queue
import threading
import subprocess
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory, Response
from torchvision.transforms import Compose, Normalize, ToTensor

repo_root = os.path.dirname(os.path.abspath(__file__))
higan_dir = os.path.join(repo_root, "HiGAN+")
if higan_dir not in sys.path:
    sys.path.insert(0, higan_dir)

from lib.utils import yaml2config
from lib.alphabet import strLabelConverter
from networks import get_model
from networks.utils import rescale_images2
from prepare_dataset import clean_and_segment_page

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(repo_root, "uploads")
app.config['OUTPUT_FOLDER'] = os.path.join(repo_root, "output")
app.config['CUSTOM_DATA_FOLDER'] = os.path.join(repo_root, "data/my_handwriting")
app.config['CROPS_TEMP_FOLDER'] = os.path.join(repo_root, "uploads/crops_temp")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CUSTOM_DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['CROPS_TEMP_FOLDER'], exist_ok=True)

MODEL = None
CFG = None
DEVICE = None
LABEL_CONVERTER = None
ORG_TRANSFORMS = None

# Training state
TRAIN_THREAD = None
TRAIN_PROC = None
TRAIN_LOGS = []
IS_TRAINING = False

def init_model(device_str="auto", ckpt_override=None):
    global MODEL, CFG, DEVICE, LABEL_CONVERTER, ORG_TRANSFORMS
    if device_str == "auto":
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(device_str)

    prev_cwd = os.getcwd()
    os.chdir(higan_dir)
    config_path = "configs/gan_image.yml"
    ckpt_path = ckpt_override or "pretrained/deploy_HiGAN+.pth"

    CFG = yaml2config(config_path)
    CFG.device = str(DEVICE)

    print(f"Loading HiGAN+ model onto {DEVICE} from {ckpt_path}...")
    MODEL = get_model(CFG.model)(CFG, config_path)
    MODEL.load(ckpt_path, DEVICE)
    MODEL.set_mode('eval')
    os.chdir(prev_cwd)

    LABEL_CONVERTER = strLabelConverter('all')
    ORG_TRANSFORMS = Compose([ToTensor(), Normalize([0.5], [0.5])])
    print("HiGAN+ model loaded.")

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
  <title>HiGAN+ Handwriting & Fine-Tuning</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      max-width: 960px;
      margin: 20px auto;
      padding: 0 16px;
      color: #111;
      background: #fafafa;
      line-height: 1.4;
    }
    h1 { margin-bottom: 4px; font-size: 1.5rem; }
    p.sub { color: #666; margin-top: 0; margin-bottom: 20px; font-size: 0.9rem; }
    fieldset {
      background: #fff;
      border: 1px solid #ccc;
      padding: 16px;
      margin-bottom: 20px;
    }
    legend {
      font-weight: bold;
      padding: 0 6px;
      font-size: 1rem;
    }
    label {
      display: block;
      font-weight: 600;
      margin-top: 10px;
      margin-bottom: 4px;
      font-size: 0.85rem;
    }
    textarea, select, input[type="text"], input[type="number"] {
      width: 100%;
      box-sizing: border-box;
      padding: 8px;
      border: 1px solid #ccc;
      font-family: inherit;
      font-size: 0.9rem;
    }
    button {
      padding: 8px 16px;
      background: #eee;
      border: 1px solid #999;
      cursor: pointer;
      font-weight: 600;
      margin-top: 10px;
    }
    button:hover { background: #ddd; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    pre {
      background: #222;
      color: #0f0;
      padding: 12px;
      border: 1px solid #444;
      font-family: "Courier New", Courier, monospace;
      font-size: 12px;
      max-height: 250px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
    .preview-img {
      max-width: 100%;
      height: auto;
      border: 1px solid #ccc;
      margin-top: 10px;
      display: block;
      background: white;
    }
    .crop-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      max-height: 320px;
      overflow-y: auto;
      border: 1px solid #ddd;
      padding: 10px;
      background: #fdfdfd;
    }
    .crop-item {
      border: 1px solid #ccc;
      padding: 6px;
      width: 140px;
      text-align: center;
      background: white;
    }
    .crop-item img {
      max-width: 100%;
      height: 40px;
      object-fit: contain;
      border: 1px solid #eee;
    }
    .crop-item input {
      width: 100%;
      font-size: 11px;
      padding: 3px;
      margin-top: 4px;
      box-sizing: border-box;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.85rem;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 6px 10px;
      text-align: left;
    }
    th { background: #f2f2f2; }
    .status-tag {
      display: inline-block;
      padding: 2px 6px;
      background: #e2e8f0;
      font-size: 0.75rem;
      font-weight: bold;
    }
  </style>
</head>
<body>

  <h1>HiGAN+ Handwriting Synthesizer</h1>
  <p class="sub">Local Device: <strong>{{ device }}</strong> | Mode: Evaluation & Fine-Tuning</p>

  <!-- SECTION 1: SYNTHESIS -->
  <fieldset>
    <legend>1. Write / Synthesize Handwriting</legend>
    <form id="synthForm">
      <label for="synthText">Text to write (supports multiple lines):</label>
      <textarea id="synthText" rows="4">Assignment 1: Artificial Intelligence&#10;Submitted by: Hari Narayan&#10;Question 1: Explain generative adversarial networks.</textarea>

      <div style="display: flex; gap: 20px; margin-top: 10px;">
        <div style="flex: 1;">
          <label for="synthStyle">Reference Style:</label>
          <select id="synthStyle">
            <optgroup label="Custom Handwriting">
              <option value="__custom_folder__">Latest Fine-Tuned / Custom Sample</option>
            </optgroup>
            <optgroup label="Built-in IAM Styles">
              {% for s in samples %}
              <option value="{{ s.name }}" {% if s.name == 'preparation' %}selected{% endif %}>{{ s.name }}</option>
              {% endfor %}
            </optgroup>
          </select>
        </div>
        <div style="flex: 1;">
          <label for="singleWordUpload">Or upload 1 reference word image:</label>
          <input type="file" id="singleWordUpload" accept="image/*">
        </div>
      </div>

      <button type="submit" id="synthBtn">Generate Handwriting</button>
    </form>

    <div id="previewArea" style="margin-top: 15px; display: none;">
      <strong>Generated Output:</strong>
      <img id="previewImg" class="preview-img" src="" alt="Handwriting Output">
      <div style="margin-top: 8px;">
        <a id="downloadLink" href="" download="handwriting.png"><button type="button">Download Image</button></a>
        <a id="tabLink" href="" target="_blank"><button type="button">Open in New Tab</button></a>
      </div>
    </div>
  </fieldset>

  <!-- SECTION 2: FINE-TUNING PIPELINE -->
  <fieldset>
    <legend>2. Fine-Tune on Your Own Handwriting</legend>

    <p style="font-size: 0.85rem; color: #444;">
      Upload a photo/scan of an A4 page with your handwriting. The system cleans shadows, cuts out each word, and lets you verify labels before training.
    </p>

    <!-- Step A: Upload & Segment -->
    <div style="background: #f9f9f9; padding: 12px; border: 1px solid #ddd; margin-bottom: 15px;">
      <strong>Step A: Segment Words from Page Photo</strong>
      <div style="margin-top: 8px;">
        <input type="file" id="pagePhotoInput" accept="image/*">
        <button type="button" id="segmentBtn">Extract & Auto-Crop Words</button>
      </div>

      <div id="cropsContainer" style="display: none; margin-top: 12px;">
        <p style="font-size: 0.8rem; color: #555; margin-bottom: 4px;">
          Extracted words detected below. Edit the label under each word if needed, then click "Save to Dataset":
        </p>
        <div class="crop-grid" id="cropGrid"></div>
        <button type="button" id="saveDatasetBtn" style="margin-top: 10px; background: #dbeafe; border-color: #93c5fd;">
          Save Labeled Words into Dataset
        </button>
        <span id="saveStatus" style="font-size: 0.85rem; margin-left: 10px; font-weight: bold;"></span>
      </div>
    </div>

    <!-- Step B: Run Training with Live Output -->
    <div style="background: #f9f9f9; padding: 12px; border: 1px solid #ddd;">
      <strong>Step B: Run Fine-Tuning</strong>
      <div style="display: flex; gap: 15px; align-items: flex-end; margin-top: 8px;">
        <div style="width: 120px;">
          <label for="epochsInput" style="margin-top:0;">Epochs:</label>
          <input type="number" id="epochsInput" value="30" min="5" max="200">
        </div>
        <div>
          <button type="button" id="startTrainBtn" style="background: #e0f2fe; border-color: #38bdf8;">
            Start Fine-Tuning
          </button>
          <button type="button" id="stopTrainBtn" disabled>Stop Training</button>
        </div>
      </div>

      <label style="margin-top: 12px;">Real-Time Training Output & Debug Logs:</label>
      <pre id="trainLogs">Waiting to start fine-tuning...</pre>
    </div>
  </fieldset>

  <!-- SECTION 3: GENERATED FILES -->
  <fieldset>
    <legend>3. Previously Generated Files</legend>
    <table>
      <thead>
        <tr>
          <th style="width: 140px;">Preview</th>
          <th>Filename</th>
          <th style="width: 100px;">Action</th>
        </tr>
      </thead>
      <tbody id="galleryTable">
        {% for g in gallery %}
        <tr>
          <td><img src="/output/{{ g.filename }}" style="height: 35px; max-width: 120px; object-fit: contain; background: white; border: 1px solid #ccc;"></td>
          <td>{{ g.filename }}</td>
          <td><a href="/output/{{ g.filename }}" download><button type="button" style="margin: 0; padding: 4px 8px; font-size: 11px;">Download</button></a></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </fieldset>

  <script>
    // SYNTHESIS FORM
    document.getElementById('synthForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('synthBtn');
      const text = document.getElementById('synthText').value.trim();
      const style = document.getElementById('synthStyle').value;
      const fileInput = document.getElementById('singleWordUpload');

      if (!text) { alert('Please enter some text.'); return; }
      btn.disabled = true;
      btn.textContent = 'Generating...';

      const formData = new FormData();
      formData.append('text', text);
      formData.append('style', style);
      if (fileInput.files.length > 0) {
        formData.append('custom_file', fileInput.files[0]);
      }

      try {
        const resp = await fetch('/api/generate', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.success) {
          const previewArea = document.getElementById('previewArea');
          const previewImg = document.getElementById('previewImg');
          const cacheBust = '?t=' + Date.now();
          previewImg.src = data.url + cacheBust;
          document.getElementById('downloadLink').href = data.url;
          document.getElementById('tabLink').href = data.url;
          previewArea.style.display = 'block';

          // Add to table
          const tbody = document.getElementById('galleryTable');
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><img src="${data.url + cacheBust}" style="height: 35px; max-width: 120px; object-fit: contain; background: white; border: 1px solid #ccc;"></td>
            <td>${data.filename}</td>
            <td><a href="${data.url}" download><button type="button" style="margin: 0; padding: 4px 8px; font-size: 11px;">Download</button></a></td>
          `;
          tbody.insertBefore(tr, tbody.firstChild);
        } else {
          alert('Error: ' + data.error);
        }
      } catch (err) {
        alert('Request failed: ' + err);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Handwriting';
      }
    });

    // SEGMENT A4 PAGE
    document.getElementById('segmentBtn').addEventListener('click', async () => {
      const fileInput = document.getElementById('pagePhotoInput');
      if (!fileInput.files.length) {
        alert('Please choose a photo of your handwriting first.');
        return;
      }
      const btn = document.getElementById('segmentBtn');
      btn.disabled = true;
      btn.textContent = 'Cleaning & Segmenting Words...';

      const formData = new FormData();
      formData.append('page_image', fileInput.files[0]);

      try {
        const resp = await fetch('/api/dataset/segment', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.success) {
          const container = document.getElementById('cropsContainer');
          const grid = document.getElementById('cropGrid');
          grid.innerHTML = '';
          data.crops.forEach((crop, idx) => {
            const div = document.createElement('div');
            div.className = 'crop-item';
            div.innerHTML = `
              <img src="/uploads/crops_temp/${crop.filename}?t=${Date.now()}">
              <input type="text" data-file="${crop.filename}" value="${crop.suggested_label}" placeholder="word label">
            `;
            grid.appendChild(div);
          });
          container.style.display = 'block';
        } else {
          alert('Segmentation error: ' + data.error);
        }
      } catch (err) {
        alert('Request failed: ' + err);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Extract & Auto-Crop Words';
      }
    });

    // SAVE DATASET LABELS
    document.getElementById('saveDatasetBtn').addEventListener('click', async () => {
      const inputs = document.querySelectorAll('#cropGrid input');
      const items = [];
      inputs.forEach(inp => {
        const val = inp.value.trim();
        if (val) {
          items.push({ filename: inp.dataset.file, label: val });
        }
      });

      if (!items.length) {
        alert('No labeled items to save.');
        return;
      }

      const status = document.getElementById('saveStatus');
      status.textContent = 'Saving dataset...';

      try {
        const resp = await fetch('/api/dataset/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ items })
        });
        const data = await resp.json();
        if (data.success) {
          status.textContent = `Saved ${data.count} word images to data/my_handwriting!`;
          status.style.color = 'green';
        } else {
          status.textContent = 'Error: ' + data.error;
          status.style.color = 'red';
        }
      } catch (err) {
        status.textContent = 'Save failed: ' + err;
        status.style.color = 'red';
      }
    });

    // FINE-TUNING STREAM & CONTROL
    let logEventSource = null;

    document.getElementById('startTrainBtn').addEventListener('click', async () => {
      const epochs = document.getElementById('epochsInput').value;
      const btn = document.getElementById('startTrainBtn');
      const stopBtn = document.getElementById('stopTrainBtn');
      const logsPre = document.getElementById('trainLogs');

      btn.disabled = true;
      stopBtn.disabled = false;
      logsPre.textContent = 'Initializing fine-tuning process...\\n';

      try {
        const resp = await fetch('/api/finetune/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ epochs: parseInt(epochs) })
        });
        const data = await resp.json();
        if (!data.success) {
          alert('Could not start training: ' + data.error);
          btn.disabled = false;
          stopBtn.disabled = true;
          return;
        }

        // Connect SSE stream
        if (logEventSource) logEventSource.close();
        logEventSource = new EventSource('/api/finetune/stream');

        logEventSource.onmessage = (e) => {
          logsPre.textContent += e.data + '\\n';
          logsPre.scrollTop = logsPre.scrollHeight;
          if (e.data.includes('TRAINING_COMPLETE') || e.data.includes('TRAINING_STOPPED')) {
            btn.disabled = false;
            stopBtn.disabled = true;
            logEventSource.close();
          }
        };

        logEventSource.onerror = () => {
          // SSE closed or finished
        };

      } catch (err) {
        alert('Start training request failed: ' + err);
        btn.disabled = false;
        stopBtn.disabled = true;
      }
    });

    document.getElementById('stopTrainBtn').addEventListener('click', async () => {
      await fetch('/api/finetune/stop', { method: 'POST' });
      document.getElementById('stopTrainBtn').disabled = true;
      document.getElementById('startTrainBtn').disabled = false;
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

    out_files = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['OUTPUT_FOLDER'], x)), reverse=True)
        out_files = [{"filename": f} for f in files]

    return render_template_string(HTML_TEMPLATE, samples=samples, gallery=out_files, device=str(DEVICE))

@app.route('/uploads/crops_temp/<filename>')
def serve_crop(filename):
    return send_from_directory(app.config['CROPS_TEMP_FOLDER'], filename)

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    text = request.form.get('text', '').strip()
    style = request.form.get('style', 'preparation').strip()

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    if 'custom_file' in request.files and request.files['custom_file'].filename:
        uploaded_file = request.files['custom_file']
        ext = os.path.splitext(uploaded_file.filename)[1] or '.png'
        custom_fn = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], custom_fn)
        uploaded_file.save(style_path)
    elif style == "__custom_folder__":
        # Use first image in data/my_handwriting
        custom_files = [f for f in os.listdir(app.config['CUSTOM_DATA_FOLDER']) if f.endswith(('.png', '.jpg'))]
        if not custom_files:
            return jsonify({"success": False, "error": "No images found in data/my_handwriting. Please upload/segment samples first."}), 400
        style_path = os.path.join(app.config['CUSTOM_DATA_FOLDER'], custom_files[0])
    else:
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

@app.route('/api/dataset/segment', methods=['POST'])
def api_segment():
    if 'page_image' not in request.files or not request.files['page_image'].filename:
        return jsonify({"success": False, "error": "No page image uploaded"}), 400

    uploaded_file = request.files['page_image']
    ext = os.path.splitext(uploaded_file.filename)[1] or '.png'
    temp_page_path = os.path.join(app.config['UPLOAD_FOLDER'], f"page_{int(time.time())}{ext}")
    uploaded_file.save(temp_page_path)

    try:
        crops = clean_and_segment_page(temp_page_path, app.config['CROPS_TEMP_FOLDER'])
        crop_items = []
        for fp in crops:
            fn = os.path.basename(fp)
            crop_items.append({"filename": fn, "suggested_label": ""})

        return jsonify({"success": True, "crops": crop_items})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/dataset/save', methods=['POST'])
def api_save_dataset():
    data = request.json or {}
    items = data.get('items', [])
    if not items:
        return jsonify({"success": False, "error": "No items provided"}), 400

    count = 0
    for it in items:
        fn = it.get('filename')
        label = it.get('label', '').strip()
        if not fn or not label:
            continue
        src = os.path.join(app.config['CROPS_TEMP_FOLDER'], fn)
        if os.path.exists(src):
            # Clean label
            clean_label = "".join([c for c in label if c.isalnum() or c in ['-', '_']])
            dst_fn = f"{clean_label}.png"
            dst_fp = os.path.join(app.config['CUSTOM_DATA_FOLDER'], dst_fn)
            # If exists, add index
            i = 1
            while os.path.exists(dst_fp):
                dst_fn = f"{clean_label}_{i}.png"
                dst_fp = os.path.join(app.config['CUSTOM_DATA_FOLDER'], dst_fn)
                i += 1
            # Copy file
            img = cv2.imread(src)
            cv2.imwrite(dst_fp, img)
            count += 1

    return jsonify({"success": True, "count": count})

def train_worker(epochs):
    global IS_TRAINING, TRAIN_PROC
    IS_TRAINING = True
    TRAIN_LOGS.clear()
    TRAIN_LOGS.append(f"Starting fine-tuning for {epochs} epochs...")

    prev_cwd = os.getcwd()
    os.chdir(higan_dir)

    # Update epochs in finetune_custom.yml if needed
    cmd = [
        os.path.join(repo_root, ".venv/bin/python"),
        "train.py",
        "--config", "./configs/finetune_custom.yml"
    ]

    try:
        TRAIN_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(TRAIN_PROC.stdout.readline, ''):
            if not line:
                break
            line_str = line.strip()
            TRAIN_LOGS.append(line_str)
            print(f"[TRAIN] {line_str}")

        TRAIN_PROC.stdout.close()
        return_code = TRAIN_PROC.wait()
        if return_code == 0:
            TRAIN_LOGS.append(">>> TRAINING_COMPLETE: Model successfully fine-tuned!")
        else:
            TRAIN_LOGS.append(f">>> TRAINING_FAILED: Process exited with code {return_code}")
    except Exception as e:
        TRAIN_LOGS.append(f">>> ERROR: {str(e)}")
    finally:
        os.chdir(prev_cwd)
        IS_TRAINING = False

@app.route('/api/finetune/start', methods=['POST'])
def api_start_train():
    global TRAIN_THREAD, IS_TRAINING
    if IS_TRAINING:
        return jsonify({"success": False, "error": "Training is already running"}), 400

    data = request.json or {}
    epochs = data.get('epochs', 30)

    # Check if custom data exists
    custom_files = [f for f in os.listdir(app.config['CUSTOM_DATA_FOLDER']) if f.endswith('.png')]
    if not custom_files:
        return jsonify({"success": False, "error": "No training images found in data/my_handwriting. Please extract/save words first!"}), 400

    TRAIN_THREAD = threading.Thread(target=train_worker, args=(epochs,))
    TRAIN_THREAD.daemon = True
    TRAIN_THREAD.start()

    return jsonify({"success": True})

@app.route('/api/finetune/stop', methods=['POST'])
def api_stop_train():
    global TRAIN_PROC, IS_TRAINING
    if TRAIN_PROC and IS_TRAINING:
        TRAIN_PROC.terminate()
        IS_TRAINING = False
        TRAIN_LOGS.append(">>> TRAINING_STOPPED by user.")
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No training in progress"})

@app.route('/api/finetune/stream')
def api_stream_logs():
    def event_stream():
        last_idx = 0
        while True:
            if last_idx < len(TRAIN_LOGS):
                while last_idx < len(TRAIN_LOGS):
                    msg = TRAIN_LOGS[last_idx]
                    last_idx += 1
                    yield f"data: {msg}\n\n"
            else:
                if not IS_TRAINING and last_idx >= len(TRAIN_LOGS):
                    break
                time.sleep(0.5)
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    init_model(args.device)
    print(f"\n🚀 HiGAN+ Simple Web UI started at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
