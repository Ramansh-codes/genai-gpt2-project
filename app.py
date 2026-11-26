"""Local web frontend server for the GPT-2 From Scratch project."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gpt2_from_scratch.runtime import configure_runtime

configure_runtime()

import tiktoken
import torch

from gpt2_from_scratch.checkpointing import load_model_state_compatible
from gpt2_from_scratch.classification import classify_sms, configure_classifier_head
from gpt2_from_scratch.config import (
    GPT_CONFIG_124M,
    TINY_GPT_CONFIG,
    get_model_config,
    model_size_from_name,
)
from gpt2_from_scratch.generation import generate, text_to_token_ids, token_ids_to_text
from gpt2_from_scratch.gpt2_weights import download_and_load_gpt2, load_weights_into_gpt
from gpt2_from_scratch.model import GPTModel, count_parameters

WEB_DIR = ROOT / "web"
TOKENIZER = tiktoken.get_encoding("gpt2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, tuple[GPTModel, dict[str, Any], int]] = {}
_CLASSIFIER_CACHE: tuple[GPTModel, int] | None = None


def _get_generation_model(model_key: str) -> tuple[GPTModel, dict[str, Any], int]:
    """Load and cache the requested GPT model."""
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    if model_key == "tiny":
        config = TINY_GPT_CONFIG.copy()
        torch.manual_seed(123)
        model = GPTModel(config).to(DEVICE)
        model.eval()
        entry = (model, config, count_parameters(model))
        _MODEL_CACHE[model_key] = entry
        return entry

    if model_key == "pretrained-124m":
        model_name = "gpt2-small (124M)"
        config = get_model_config(model_name, drop_rate=0.0, qkv_bias=True)
        _, params = download_and_load_gpt2(
            model_size_from_name(model_name),
            models_dir=ROOT / "gpt2",
        )
        model = GPTModel(config)
        load_weights_into_gpt(model, params)
        model.to(DEVICE)
        model.eval()
        entry = (model, config, count_parameters(model))
        _MODEL_CACHE[model_key] = entry
        return entry

    if model_key == "sft-124m":
        ckpt = ROOT / "gpt2-small124M-sft.pth"
        if not ckpt.exists():
            raise FileNotFoundError(
                "SFT checkpoint not found (gpt2-small124M-sft.pth). "
                "Run `python scripts\\ch07_instruction_tuning_eval.py --mode train` first."
            )
        config = get_model_config("gpt2-small (124M)", drop_rate=0.0, qkv_bias=True)
        model = GPTModel(config)
        load_model_state_compatible(model, ckpt, DEVICE)
        model.to(DEVICE)
        model.eval()
        entry = (model, config, count_parameters(model))
        _MODEL_CACHE[model_key] = entry
        return entry

    raise ValueError(f"Unsupported model key: {model_key}")


def _get_classifier_model() -> tuple[GPTModel, int]:
    """Load and cache the SMS spam classifier checkpoint."""
    global _CLASSIFIER_CACHE
    if _CLASSIFIER_CACHE is not None:
        return _CLASSIFIER_CACHE

    ckpt = ROOT / "review_classifier.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            "Classifier checkpoint not found (review_classifier.pth). "
            "Run `python scripts\\ch06_finetune_classifier.py --mode train` first."
        )

    config = get_model_config("gpt2-small (124M)", drop_rate=0.0, qkv_bias=True)
    model = GPTModel(config)
    model = configure_classifier_head(model, emb_dim=config["emb_dim"])
    load_model_state_compatible(model, ckpt, DEVICE)
    model.to(DEVICE)
    model.eval()
    _CLASSIFIER_CACHE = (model, 120)
    return _CLASSIFIER_CACHE


def _status_payload() -> dict[str, Any]:
    pretrained_dir = ROOT / "gpt2" / "124M"
    has_pretrained = (pretrained_dir / "model.ckpt.data-00000-of-00001").exists()
    has_sft = (ROOT / "gpt2-small124M-sft.pth").exists()
    has_classifier = (ROOT / "review_classifier.pth").exists()

    return {
        "device": str(DEVICE),
        "loaded_models": list(_MODEL_CACHE.keys()),
        "models": [
            {
                "id": "pretrained-124m",
                "label": "GPT-2 Small (124M Pretrained)",
                "available": has_pretrained,
                "params": 163_009_536,
                "context_length": GPT_CONFIG_124M["context_length"],
                "note": "Uses local OpenAI GPT-2 124M weights" if has_pretrained else "Requires downloading gpt2/124M weights",
            },
            {
                "id": "tiny",
                "label": "Tiny GPT (From Scratch)",
                "available": True,
                "params": 13_270_016,
                "context_length": TINY_GPT_CONFIG["context_length"],
                "note": "Untrained 2-layer architecture demo",
            },
            {
                "id": "sft-124m",
                "label": "GPT-2 Small (Instruction SFT)",
                "available": has_sft,
                "params": 163_009_536,
                "context_length": GPT_CONFIG_124M["context_length"],
                "note": "Ready" if has_sft else "Train via ch07_instruction_tuning_eval.py --mode train",
            },
        ],
        "classifier_available": has_classifier,
        "configs": {
            "tiny": TINY_GPT_CONFIG,
            "124m": GPT_CONFIG_124M,
        },
    }


class AppHandler(BaseHTTPRequestHandler):
    """Serve static frontend files and JSON inference APIs."""

    server_version = "GPT2FromScratchWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html_path = WEB_DIR / "index.html"
            if not html_path.exists():
                self._send_json({"error": "web/index.html not found"}, HTTPStatus.NOT_FOUND)
                return
            content = html_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if self.path == "/api/status":
            self._send_json(_status_payload())
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            data = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/tokenize":
            text = str(data.get("text", ""))
            token_ids = TOKENIZER.encode(text, allowed_special={"<|endoftext|>"})
            tokens = [
                {
                    "id": tid,
                    "text": TOKENIZER.decode([tid]),
                }
                for tid in token_ids
            ]
            self._send_json(
                {
                    "count": len(token_ids),
                    "char_count": len(text),
                    "tokens": tokens,
                }
            )
            return

        if self.path == "/api/generate":
            prompt = str(data.get("prompt", ""))
            if not prompt.strip():
                self._send_json({"error": "Prompt cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            model_key = str(data.get("model", "pretrained-124m"))
            max_new_tokens = max(1, min(int(data.get("max_new_tokens", 40)), 256))
            temperature = max(0.0, min(float(data.get("temperature", 0.7)), 2.0))
            raw_top_k = data.get("top_k", 50)
            top_k = int(raw_top_k) if raw_top_k not in (None, "", 0, "0") else None
            if top_k is not None and top_k <= 0:
                top_k = None

            try:
                t0 = time.perf_counter()
                with _MODEL_LOCK:
                    model, config, param_count = _get_generation_model(model_key)
                    input_ids = text_to_token_ids(prompt, TOKENIZER).to(DEVICE)
                    prompt_len = input_ids.shape[1]
                    if prompt_len > config["context_length"]:
                        input_ids = input_ids[:, -config["context_length"] :]
                        prompt_len = input_ids.shape[1]

                    out_ids = generate(
                        model=model,
                        idx=input_ids,
                        max_new_tokens=max_new_tokens,
                        context_size=config["context_length"],
                        temperature=temperature,
                        top_k=top_k,
                        eos_id=50256,
                    )
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                new_tokens = out_ids[0, prompt_len:]
                generated_text = TOKENIZER.decode(new_tokens.tolist())
                full_text = token_ids_to_text(out_ids, TOKENIZER)
                self._send_json(
                    {
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "full_text": full_text,
                        "prompt_tokens": prompt_len,
                        "tokens_generated": int(new_tokens.numel()),
                        "elapsed_ms": elapsed_ms,
                        "params": param_count,
                        "device": str(DEVICE),
                        "model": model_key,
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/classify":
            text = str(data.get("text", "")).strip()
            if not text:
                self._send_json({"error": "Message text cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return
            try:
                t0 = time.perf_counter()
                with _MODEL_LOCK:
                    model, max_length = _get_classifier_model()
                    label = classify_sms(text, model, TOKENIZER, DEVICE, max_length)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                self._send_json(
                    {
                        "text": text,
                        "label": label,
                        "elapsed_ms": elapsed_ms,
                        "device": str(DEVICE),
                    }
                )
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPT-2 From Scratch web frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload GPT-2 124M weights into memory before serving.",
    )
    args = parser.parse_args()

    if args.preload:
        print("Preloading GPT-2 124M model into memory...")
        with _MODEL_LOCK:
            _get_generation_model("pretrained-124m")
        print("Preload complete.")

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"GPT-2 From Scratch UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()

