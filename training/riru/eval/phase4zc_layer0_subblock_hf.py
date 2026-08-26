"""Phase 4ZC Section15: layer0 sub-block(RMSNorm/attention-residual/FFN-norm/FFN-output)の
HF側キャプチャ。llama.cppのcb()命名(attn_norm-0/ffn_inp-0/ffn_norm-0/ffn_out-0)と対応させる。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

from phase4zc_hf_hidden_dump import MODEL_PATH, build_full_text  # noqa: E402


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    _, full_text = build_full_text(tokenizer)
    encoded = tokenizer(full_text, return_tensors="pt").to(model.device)

    layer0 = model.model.layers[0]
    captured = {}

    def _hook(name):
        def _fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[name] = hs.detach()[0, -1, :].float().cpu().contiguous()
        return _fn

    def _pre_hook_ffn_inp(module, args, kwargs):
        # post_attention_layernorm への入力 = ffn_inp (attn出力+residual)
        x = args[0] if args else kwargs["hidden_states"]
        captured["ffn_inp-0"] = x.detach()[0, -1, :].float().cpu().contiguous()

    handles = [
        layer0.input_layernorm.register_forward_hook(_hook("attn_norm-0")),
        layer0.post_attention_layernorm.register_forward_pre_hook(_pre_hook_ffn_inp, with_kwargs=True),
        layer0.post_attention_layernorm.register_forward_hook(_hook("ffn_norm-0")),
        layer0.mlp.register_forward_hook(_hook("ffn_out-0")),
    ]

    with torch.no_grad():
        model(**encoded, use_cache=False)

    for h in handles:
        h.remove()

    out_path = REPORTS_DIR / "phase4zc_hf_layer0_subblock.safetensors"
    save_file(captured, str(out_path))
    print(f"Saved -> {out_path}")
    for k, v in captured.items():
        print(k, v.shape, float(v.norm()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
