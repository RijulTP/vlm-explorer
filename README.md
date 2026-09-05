# VLM Explorer

An educational web app that visualizes how CLIP (Contrastive Language-Image Pre-training) connects images and text — step by step, not as a black box.

## What it does

Upload an image and type a caption, then watch CLIP's internal pipeline unfold:

1. **Input** — Raw image and text query
2. **Patchify** — The image is split into a 7×7 grid of 32×32 patches
3. **Vision Encoding** — Each patch becomes a 768-dim vector through 12 transformer layers
4. **Text Encoding** — Your text is tokenized and encoded into the same vector space
5. **Cross-Modal Alignment** — Cosine similarity between image and text embeddings
6. **Attention Overlay** — A heatmap showing which patches match the text

There's also a **Zero-Shot Classification** mode: give it an image and a comma-separated list of labels, and it ranks them by similarity — no training needed.

## Setup

### Prerequisites
- Python 3.9+
- pip

### 1. Create a virtual environment

```bash
cd vlm-explorer
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the CLIP model

The model weights (~580MB) need to be downloaded once. Use `hf` CLI with resumable transfer:

```bash
pip install hf_transfer
hf download openai/clip-vit-base-patch32 --local-dir ./model_cache/clip-vit-base-patch32
```

If the download drops, just rerun the same command — it resumes from where it left off.

### 4. Run the app

```bash
TOKENIZERS_PARALLELISM=false python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Sample images

Three sample images are included in `sample_images/` so the demo works out of the box:
- `dog.jpg` — A dog
- `cat.jpg` — A cat
- `car.jpg` — A car

Click a thumbnail in the UI to load it instantly, or upload your own.

## Tech stack

- **Backend**: Python, FastAPI, `transformers`, `torch`, Pillow, matplotlib
- **Frontend**: Single-page HTML/CSS/JS — no build tools, no frameworks
- **Model**: `openai/clip-vit-base-patch32` via Hugging Face (runs on CPU, fully offline after download)

## Project structure

```
vlm-explorer/
├── backend/
│   └── app.py              # FastAPI server + CLIP pipeline logic
├── frontend/
│   └── index.html          # Single-page app (HTML/CSS/JS)
├── model_cache/
│   └── clip-vit-base-patch32/  # Local CLIP model weights
├── sample_images/
│   ├── dog.jpg
│   ├── cat.jpg
│   └── car.jpg
├── requirements.txt
└── README.md
```

## How the pipeline works (for the curious)

### Patchify
CLIP's Vision Transformer doesn't see raw pixels. It resizes the image to 224×224 and splits it into a 7×7 grid of 32×32 patches (49 patches total). Each patch is flattened into a vector — that's the "vision tokenization" step.

### Vision Encoding
The 49 patch vectors + a learned CLS token are fed through 12 transformer encoder layers with self-attention. The CLS token's final hidden state is projected through a linear layer into CLIP's shared embedding space.

### Text Encoding
Your text is tokenized using CLIP's BPE tokenizer, then run through a 12-layer text transformer. The final EOS token embedding is projected into the same space as the image.

### Cross-Modal Alignment
Cosine similarity between the image and text embeddings gives a single score: how "close" they are in the shared vector space. During training, CLIP learned to push matching image-text pairs together and non-matching pairs apart.

### Attention Overlay
Per-patch relevance is computed by taking the cosine similarity between each patch's projected embedding and the text embedding. This creates a spatial heatmap showing where in the image the model finds the text-relevant content.
