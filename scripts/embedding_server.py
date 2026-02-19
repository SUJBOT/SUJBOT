#!/usr/bin/env python3
"""
OpenAI-compatible embedding server for Qwen3-VL-Embedding-8B.

Serves /v1/embeddings endpoint matching OpenAI API format.
Handles both text and image (data URI) inputs.
Designed to run inside scitrera/dgx-spark-vllm Docker container on GB10.

Usage:
    python3 embedding_server.py [--port 8081] [--model Qwen/Qwen3-VL-Embedding-8B]
"""

import argparse
import base64
import io
import logging
import time
from typing import List, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Server")

# Global model state
model = None
processor = None
model_name = ""

# Qwen3-VL-Embedding requires a system instruction for proper embedding alignment.
# See: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
DEFAULT_INSTRUCTION = "Represent the user's input."


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = ""
    dimensions: int = Field(default=0, description="Not used, always returns full dimensions")
    encoding_format: str = Field(default="float", description="Encoding format (only float supported)")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@app.get("/v1/models")
def list_models():
    return ModelList(data=[ModelInfo(id=model_name)])


@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        embeddings = encode_inputs(inputs)
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    data = [
        EmbeddingData(index=i, embedding=emb.tolist())
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=model_name,
        usage={
            "prompt_tokens": len(inputs),
            "total_tokens": len(inputs),
        },
    )


def _is_data_uri(s: str) -> bool:
    """Check if string is a data URI (base64-encoded image)."""
    return s.startswith("data:")


def _decode_data_uri(data_uri: str) -> Image.Image:
    """Decode a data URI to a PIL Image."""
    # Strip "data:image/png;base64," prefix
    if ";base64," in data_uri:
        b64_data = data_uri.split(";base64,", 1)[1]
    else:
        raise ValueError(f"Unsupported data URI format: {data_uri[:50]}...")

    img_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _pooling_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-token pooling: extract hidden state at the last valid token position.

    Matches the official Qwen3VLEmbedder._pooling_last() implementation.
    The last token has attended to all previous tokens via causal attention,
    making it the natural summary representation for decoder-only models.
    """
    flipped = attention_mask.flip(dims=[1])
    last_one_positions = flipped.argmax(dim=1)
    col = attention_mask.shape[1] - last_one_positions - 1
    row = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[row, col]


def _embed_single(content) -> np.ndarray:
    """Embed a single input (text string or PIL Image).

    Returns L2-normalized embedding as numpy array.
    """
    if isinstance(content, Image.Image):
        user_content = [{"type": "image", "image": content}]
    else:
        user_content = [{"type": "text", "text": content}]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
        {"role": "user", "content": user_content},
    ]

    processed = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    )
    processed = {k: v.to(model.device) for k, v in processed.items()}

    with torch.no_grad():
        outputs = model(**processed)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        attention_mask = processed.get(
            "attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)
        )
        embedding = _pooling_last(hidden, attention_mask)

    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu().float().numpy()


def encode_inputs(inputs: List[str]) -> List[np.ndarray]:
    """Encode a list of inputs (text or data URIs) into normalized embeddings.

    Text inputs are batched together for efficiency.
    Image inputs are processed one at a time (variable pixel dimensions).
    """
    # Separate text and image inputs, keeping track of original indices
    text_indices = []
    text_strings = []
    image_indices = []
    image_pils = []

    for i, inp in enumerate(inputs):
        if _is_data_uri(inp):
            image_indices.append(i)
            image_pils.append(_decode_data_uri(inp))
        else:
            text_indices.append(i)
            text_strings.append(inp)

    results = [None] * len(inputs)

    # Batch-encode text inputs (more efficient)
    if text_strings:
        text_embeddings = _encode_text_batch(text_strings)
        for idx, emb in zip(text_indices, text_embeddings):
            results[idx] = emb

    # Encode images one by one (variable sizes prevent batching)
    for idx, pil_img in zip(image_indices, image_pils):
        results[idx] = _embed_single(pil_img)

    return results


def _encode_text_batch(texts: List[str]) -> List[np.ndarray]:
    """Batch-encode text strings into normalized embeddings."""
    messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        for text in texts
    ]

    processed = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    )
    processed = {k: v.to(model.device) for k, v in processed.items()}

    with torch.no_grad():
        outputs = model(**processed)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        attention_mask = processed.get(
            "attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)
        )
        embeddings = _pooling_last(hidden, attention_mask)

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return [emb.cpu().float().numpy() for emb in embeddings]


def load_model(name: str):
    global model, processor, model_name
    model_name = name

    logger.info(f"Loading model: {name}")
    start = time.time()

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, padding_side="right")
    # Load full Qwen3VLForConditionalGeneration (checkpoint keys match this class).
    # AutoModel loads Qwen3VLModel which misaligns weight keys → random language model.
    full_model = AutoModelForImageTextToText.from_pretrained(
        name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Extract the base Qwen3VLModel (no LM head needed for embeddings)
    model = full_model.model
    model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info(f"Model on GPU: {torch.cuda.get_device_name()}")

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    # Test text embedding
    test_emb = _encode_text_batch(["test"])
    logger.info(f"Text embedding dimensions: {len(test_emb[0])}")

    # Test image embedding (small 64x64 test image)
    test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    test_img_emb = _embed_single(test_img)
    logger.info(f"Image embedding dimensions: {len(test_img_emb)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-8B")
    args = parser.parse_args()

    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
