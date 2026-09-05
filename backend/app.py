"""
VLM Explorer Backend — FastAPI server that exposes CLIP's internal pipeline step-by-step.

This server loads the openai/clip-vit-base-patch32 model and provides endpoints
that run each stage of the vision-language pipeline, returning intermediate
outputs so the frontend can visualize them.
"""

import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from transformers import CLIPModel, CLIPProcessor
import torch

app = FastAPI(title="VLM Explorer")

# ---------------------------------------------------------------------------
# Model loading — runs once at startup
# ---------------------------------------------------------------------------
import os
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache", "clip-vit-base-patch32")
print(f"Loading CLIP model from {_MODEL_DIR}...")
model = CLIPModel.from_pretrained(_MODEL_DIR, local_files_only=True)
processor = CLIPProcessor.from_pretrained(_MODEL_DIR, local_files_only=True)
model.eval()
print("Model loaded.")


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded data-URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def tensor_to_list(t: torch.Tensor) -> list:
    """Detach a tensor and return it as a plain Python list."""
    return t.detach().cpu().tolist()


# ---------------------------------------------------------------------------
# POST /pipeline — run the full pipeline and return every intermediate result
# ---------------------------------------------------------------------------
@app.post("/pipeline")
async def run_pipeline(
    image: UploadFile = File(...),
    text: str = Form("a photo"),
):
    # ------------------------------------------------------------------
    # Stage 0 — Raw input
    # ------------------------------------------------------------------
    raw_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    input_b64 = pil_to_base64(raw_image)

    # ------------------------------------------------------------------
    # Stage 1 — Patchify (show grid dimensions)
    # ------------------------------------------------------------------
    # CLIP ViT-B/32 uses 32×32 patches on a 224×224 image → 7×7 grid
    patch_size = 32
    image_size = 224
    grid_h = image_size // patch_size  # 7
    grid_w = image_size // patch_size  # 7

    # Create a grid-overlaid version of the image for display
    resized = raw_image.resize((image_size, image_size))
    grid_img = resized.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(grid_img)
    for x in range(0, image_size + 1, patch_size):
        draw.line([(x, 0), (x, image_size)], fill="cyan", width=1)
    for y in range(0, image_size + 1, patch_size):
        draw.line([(0, y), (image_size, y)], fill="cyan", width=1)
    patchify_b64 = pil_to_base64(grid_img)

    # ------------------------------------------------------------------
    # Stage 2 — Vision encoding (patch features + pooled embedding)
    # ------------------------------------------------------------------
    inputs = processor(
        text=[text],
        images=resized,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        vision_out = model.vision_model(
            inputs["pixel_values"],
            output_hidden_states=True,
            return_dict=True,
        )

    # Last hidden state: shape [1, num_patches + 1 (CLS), hidden_dim]
    last_hidden = vision_out.hidden_states[-1]  # [1, 50, 768] for patch32
    patch_features = last_hidden[:, 1:, :]  # drop CLS token → [1, 49, 768]
    cls_embedding = last_hidden[:, 0, :]    # CLS token → [1, 768]

    # Project through CLIP's visual projection to get the final image embedding
    image_embeds = model.visual_projection(cls_embedding)
    image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    # Per-patch magnitude for heatmap (simple single-value reduction)
    patch_norms = patch_features.norm(dim=-1).squeeze(0)  # [49]
    patch_norms_np = patch_norms.numpy()
    # Normalize to 0-1
    patch_norms_np = (patch_norms_np - patch_norms_np.min()) / (
        patch_norms_np.max() - patch_norms_np.min() + 1e-8
    )
    patch_heatmap = patch_norms_np.reshape(grid_h, grid_w).tolist()

    # ------------------------------------------------------------------
    # Stage 3 — Text encoding (tokens + embedding)
    # ------------------------------------------------------------------
    with torch.no_grad():
        text_out = model.text_model(
            inputs["input_ids"],
            output_hidden_states=True,
            return_dict=True,
        )

    text_last_hidden = text_out.hidden_states[-1]  # [1, seq_len, 768]
    # Use the EOS token embedding (last non-padding token)
    eos_indices = (inputs["attention_mask"].sum(dim=1) - 1)  # [1]
    text_cls = text_last_hidden[
        torch.arange(text_last_hidden.size(0)), eos_indices
    ]
    text_embeds = model.text_projection(text_cls)
    text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Decode tokens for display
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [processor.tokenizer.decode([tid]) for tid in token_ids]
    # Filter out special tokens for cleaner display
    display_tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in ("[CLS]", "[SEP]", "<|startoftext|>", "</tool_call>")]

    # ------------------------------------------------------------------
    # Stage 4 — Cross-modal alignment (cosine similarity)
    # ------------------------------------------------------------------
    cosine_sim = torch.nn.functional.cosine_similarity(
        image_embeds_norm, text_embeds_norm
    ).item()

    # ------------------------------------------------------------------
    # Stage 5 — Patch-level attention / relevance overlay
    # ------------------------------------------------------------------
    # Approximate patch-text relevance by computing cosine similarity between
    # each patch's hidden-state vector and the text embedding (both projected).
    # Project patch features through the visual projection
    patch_features_flat = patch_features.squeeze(0)  # [49, 768]
    patch_proj = model.visual_projection(patch_features_flat)  # [49, 64] (projection dim is 64 for CLIP ViT-B/32)
    patch_proj_norm = patch_proj / patch_proj.norm(dim=-1, keepdim=True)

    # Text embedding already projected and normalized: [1, 64]
    # Compute per-patch cosine similarity with text
    relevance_scores = torch.mm(patch_proj_norm, text_embeds_norm.t()).squeeze()  # [49]
    relevance_np = relevance_scores.detach().numpy()
    # Normalize to 0-1 for visualization
    relevance_norm = (relevance_np - relevance_np.min()) / (
        relevance_np.max() - relevance_np.min() + 1e-8
    )
    relevance_grid = relevance_norm.reshape(grid_h, grid_w)

    # Create heatmap overlay on original image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=72)
    ax.imshow(resized)
    ax.imshow(relevance_grid, cmap="jet", alpha=0.6, vmin=0, vmax=1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    overlay_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return JSONResponse(
        {
            "input_image": input_b64,
            "patchify": {
                "image": patchify_b64,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "patch_size": patch_size,
                "num_patches": grid_h * grid_w,
            },
            "vision_encoding": {
                "patch_heatmap": patch_heatmap,
                "embedding_dim": image_embeds.shape[-1],
                "num_patches": patch_features.shape[1],
            },
            "text_encoding": {
                "tokens": display_tokens,
                "embedding_dim": text_embeds.shape[-1],
            },
            "alignment": {
                "cosine_similarity": round(cosine_sim, 4),
            },
            "attention_overlay": {
                "image": overlay_b64,
                "relevance_grid": relevance_grid.tolist(),
            },
        }
    )


# ---------------------------------------------------------------------------
# POST /zero-shot — zero-shot classification mode
# ---------------------------------------------------------------------------
@app.post("/zero-shot")
async def zero_shot_classify(
    image: UploadFile = File(...),
    labels: str = Form("a cat, a dog, a car"),
):
    raw_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    input_b64 = pil_to_base64(raw_image)
    resized = raw_image.resize((224, 224))

    label_list = [l.strip() for l in labels.split(",") if l.strip()]

    inputs = processor(
        text=label_list,
        images=resized,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Cosine similarities (already normalized by CLIP's forward pass)
    logits = outputs.logits_per_image[0]  # [num_labels]
    probs = logits.softmax(dim=-1).numpy().tolist()
    sims = logits.numpy().tolist()

    results = sorted(
        [
            {"label": label, "similarity": round(s, 4), "probability": round(p, 4)}
            for label, s, p in zip(label_list, sims, probs)
        ],
        key=lambda x: x["similarity"],
        reverse=True,
    )

    return JSONResponse(
        {
            "input_image": input_b64,
            "results": results,
        }
    )


# ---------------------------------------------------------------------------
# Serve the frontend as static files
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
