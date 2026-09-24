"""Model backends: the real laya checkpoint, and the stub used by ``--dry-run``.

A backend owns a tokenizer (``tok``), the checkpoint's shipped config (``cfg``) and one
``predict`` call. The runner never touches torch directly, so the stub path exercises the same
builder, the same cells, the same JSONL writer and the same metrics as the real one.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple


class BackendBase:
    name = "base"

    tok: Any = None
    cfg: Dict[str, Any] = {}
    device = "cpu"

    def shipped_defaults(self) -> Tuple[int, int]:
        return int(self.cfg.get("max_len", 512)), int(self.cfg.get("head_max_len", 192))

    def predict(self, state: str, questions: Dict[str, Any],
                max_len: Optional[int] = None, head_max_len: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def info(self) -> Dict[str, Any]:
        return {"backend": self.name}


class LayaBackend(BackendBase):
    """The real checkpoint through the public ``laya.load`` / ``Agent.predict`` API."""

    name = "laya"

    def __init__(self, checkpoint: str, device: str = "cuda", subfolder: Optional[str] = None):
        import laya

        self.checkpoint = checkpoint
        self.laya = laya
        t0 = time.perf_counter()
        self.agent = laya.load(checkpoint, device=device, subfolder=subfolder)
        self.load_seconds = time.perf_counter() - t0
        self.tok = self.agent.tok
        self.cfg = dict(self.agent.cfg)
        self.device = str(self.agent.device)
        self.dtype = str(self.agent.dtype)
        self.amp_enabled = bool(self.agent.amp_enabled)
        self.n_params = int(sum(p.numel() for p in self.agent.model.parameters()))

    def predict(self, state: str, questions: Dict[str, Any],
                max_len: Optional[int] = None, head_max_len: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        r = self.agent.predict(state, questions, max_len=max_len, head_max_len=head_max_len)
        dt = time.perf_counter() - t0
        return {"result": r, "latency_s": dt, "input_tokens": int(r["usage"]["input_tokens"])}

    def close(self) -> None:
        try:
            with self.agent:
                pass
        except Exception:
            pass

    def info(self) -> Dict[str, Any]:
        out = {
            "backend": self.name,
            "checkpoint": self.checkpoint,
            "laya_version": getattr(self.laya, "__version__", None),
            "device": self.device,
            "dtype": self.dtype,
            "amp_enabled": self.amp_enabled,
            "parameters": self.n_params,
            "load_seconds": round(self.load_seconds, 3),
            "shipped_max_len": self.cfg.get("max_len"),
            "shipped_head_max_len": self.cfg.get("head_max_len"),
            "encoder": self.cfg.get("encoder"),
            "head_layers": self.cfg.get("head_layers"),
            "amp_dtype_cfg": self.cfg.get("amp_dtype"),
            "tokenizer_class": type(self.tok).__name__,
            "tokenizer_vocab_size": getattr(self.tok, "vocab_size", None),
            "load_path_note": ("laya.Agent.__init__ builds the model under transformers' "
                               "no_init_weights() and fills it from safetensors.load_file, which "
                               "reads lazily from the file; the public loader exposes no "
                               "low_cpu_mem_usage flag."),
        }
        try:
            import torch
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                free, total = torch.cuda.mem_get_info()
                out["gpu_mem_free_bytes_after_load"] = int(free)
                out["gpu_mem_total_bytes"] = int(total)
                out["gpu_mem_allocated_bytes_after_load"] = int(torch.cuda.memory_allocated())
                out["gpu_mem_reserved_bytes_after_load"] = int(torch.cuda.memory_reserved())
        except Exception:
            pass
        return out


class StubBackend(BackendBase):
    """A tiny random BERT + the real ``Agent.predict`` path. No checkpoint, no GPU, no network."""

    name = "stub"

    def __init__(self, seed: int = 1234, max_len: int = 64, head_max_len: int = 24,
                 vocab_size: int = 4096, hidden_size: int = 32, layers: int = 2,
                 max_position_embeddings: int = 8192):
        from laya.agent import Agent
        from laya.common import DecisionModel
        from transformers import AutoConfig, AutoModel

        import torch

        self.torch = torch
        torch.manual_seed(seed)
        self.seed = seed
        agent = object.__new__(Agent)
        self._agent = agent
        agent.model_id = "stub"
        agent.cfg = {"max_len": max_len, "head_max_len": head_max_len, "encoder": "stub",
                     "head_layers": 1, "act_costs": {"escalate": 0.5}, "amp_dtype": "bf16"}
        agent.tok = StubTokenizer(vocab_size=vocab_size)
        ecfg = AutoConfig.for_model("bert", hidden_size=hidden_size, num_hidden_layers=layers,
                                    num_attention_heads=2, intermediate_size=2 * hidden_size,
                                    vocab_size=vocab_size,
                                    max_position_embeddings=max_position_embeddings)
        agent.model = DecisionModel(AutoModel.from_config(ecfg), head_layers=1, n_act=2).eval()
        agent.device = torch.device("cpu")
        agent.dtype = torch.float32
        agent.amp_enabled = False
        agent.mps_amp_min_rows = 5
        agent._fast = None
        agent.temperature = [1.0, 1.0, 1.0]
        agent.temperature_raw = [1.0, 1.0, 1.0]
        agent.temperature_by_options = {}
        agent.temperature_by_options_raw = {}
        agent.lang_temperatures = {}
        agent.hooks_raise = True
        agent.hooks_concurrent = True
        agent._hooks_lock = None
        agent._hooks_mutex = threading.Lock()
        self.agent = agent
        self.tok = agent.tok
        self.cfg = dict(agent.cfg)
        self.device = "cpu"
        self.dtype = "torch.float32"
        self.amp_enabled = False
        self.n_params = int(sum(p.numel() for p in agent.model.parameters()))

    def predict(self, state: str, questions: Dict[str, Any],
                max_len: Optional[int] = None, head_max_len: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        r = self.agent.predict(state, questions, max_len=max_len, head_max_len=head_max_len)
        dt = time.perf_counter() - t0
        return {"result": r, "latency_s": dt, "input_tokens": int(r["usage"]["input_tokens"])}

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "note": ("stub path: whitespace tokenizer + a 2-layer random BERT of the same shape "
                     "family as the real encoder; exercises builder, Agent.predict, collate, "
                     "truncation and the JSONL writer, but its predictions are random"),
            "seed": self.seed,
            "parameters": self.n_params,
            "shipped_max_len": self.cfg.get("max_len"),
            "shipped_head_max_len": self.cfg.get("head_max_len"),
            "tokenizer_class": type(self.tok).__name__,
            "tokenizer_vocab_size": getattr(self.tok, "vocab_size", None),
            "device": "cpu",
        }


class StubTokenizer:
    """Whitespace tokenizer whose ids round-trip exactly, so filler token budgets are exact.

    ``t<id>`` tokens re-tokenize to the same id, which is what lets the dry run exercise
    ``DocBuilder._filler_exact``'s decode/slice/re-tokenize loop and still assert an exact budget.
    """

    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2
    mask_token_id = 3
    mask_token = "[MASK]"

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size

    def _id(self, word: str) -> int:
        if len(word) > 1 and word[0] == "t" and word[1:].isdigit():
            return int(word[1:]) % self.vocab_size
        import zlib
        return 10 + (zlib.crc32(word.encode("utf-8")) % (self.vocab_size - 10))

    def __call__(self, text, add_special_tokens: bool = False, truncation: bool = False,
                 max_length: Optional[int] = None, return_offsets_mapping: bool = False):
        if not isinstance(text, str):
            text = str(text)
        ids, offsets = [], []
        i, n = 0, len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < n and not text[j].isspace():
                j += 1
            ids.append(self._id(text[i:j]))
            offsets.append((i, j))
            i = j
        if truncation and max_length:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids) -> str:
        return " ".join("t%d" % int(i) for i in ids)

    def __len__(self) -> int:
        return self.vocab_size
