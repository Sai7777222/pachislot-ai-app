"""Phase 4ZC Section11: HF/PyTorch側のlayer-wise hidden state抽出。

重要: これまでのPhase4Z/4ZAの「HF」条件はbitsandbytes 4bit NF4量子化ロード
(BitsAndBytesConfig load_in_4bit=True)を使用していたことが判明した(phase4z_logits_compare.py,
phase4z_identity_eval_hf.py で確認)。Phase4ZC Section11の指示は明示的に
「BF16/eval/no_grad」のフル精度forwardを要求しているため、本スクリプトは
bitsandbytes量子化を一切使わず、torch_dtype=torch.bfloat16の素のHFロードで実行する。
これにより、従来の「HF」参照値自体にbnb 4bit量子化ノイズが混入していた可能性を
Section21のprecision-sensitivity testで別途切り分ける土台とする。

E36 original(system prompt + user prompt)をchat templateでrenderし、
forced prefix「こんにちは〜！私はパチスロの専門アシスタントの」を追記した上で、
1回のforward(output_hidden_states=True, use_cache=False)を実行し、
embedding出力 + 各layer(0-47)出力 + post-final-norm + logitsの
最終トークン位置のベクトルを保存する。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MODEL_PATH = str(TRAINING_ROOT / "merged" / "riru-qwen-final-hf")
REPORTS_DIR = TRAINING_ROOT / "reports"

FORCED_PREFIX = "こんにちは〜！私はパチスロの専門アシスタントの"


def build_full_text(tokenizer) -> tuple[str, str]:
    from phase4z_probes import PROBE_SET_C

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    e36_original = PROBE_SET_C[0]["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": e36_original},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = prompt_text + FORCED_PREFIX
    return prompt_text, full_text


def main(attn_impl: str = "eager", dtype_name: str = "bfloat16", tag: str = "",
         device_map: str = "cuda:0") -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # トークンID再確認(推測ではなくtokenizerから直接取得。4ZBと同じID)。
    ri_ids = tokenizer.encode("リ", add_special_tokens=False)
    ru_ids = tokenizer.encode("ル", add_special_tokens=False)
    assert len(ri_ids) == 1 and len(ru_ids) == 1, f"unexpected multi-token: {ri_ids} {ru_ids}"
    ri_id, ru_id = ri_ids[0], ru_ids[0]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    prompt_text, full_text = build_full_text(tokenizer)
    encoded = tokenizer(full_text, return_tensors="pt").to(model.device)
    n_tokens = encoded["input_ids"].shape[1]

    # 重要な設計上の注意(発見済みのバグの回避):
    # HFのoutput_hidden_states=Trueで返るhidden_statesタプルは、ループ内で「各layerへの入力」を
    # 追記し、ループ終了後に「最終layer出力にfinal normを適用した値」を最後の要素として追記する。
    # そのため hidden_states[-1] は post-final-norm 値であり、layer47の生(pre-norm)出力ではない
    # (実測: layer47_outputとpost_final_normがbit-for-bitで完全一致することを確認して発覚)。
    # これを避けるため、各decoder layerモジュールにforward hookを直接登録し、
    # llama.cppのcb(cur, "l_out", il)と同じ意味(layer il の生の残差ストリーム出力)を
    # 確実に捕捉する。
    captured = {}

    def _make_layer_hook(idx):
        def _hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[f"layer_{idx:02d}_output"] = hs.detach()
        return _hook

    def _norm_hook(module, inp, out):
        captured["post_final_norm"] = out.detach()

    handles = []
    for idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(_make_layer_hook(idx)))
    handles.append(model.model.norm.register_forward_hook(_norm_hook))

    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True, use_cache=False)

    for h in handles:
        h.remove()

    hidden_states = out.hidden_states  # tuple length = num_layers + 1 (embedding + each layer input)
    n_layers_plus_embd = len(hidden_states)

    tensors = {}
    # embedding_output = input to layer 0 = hidden_states[0] (this element is unambiguous:
    # it precedes any layer processing, so no post-norm confusion applies here).
    tensors["embedding_output"] = hidden_states[0][0, -1, :].detach().float().cpu().contiguous()
    for idx in range(len(model.model.layers)):
        vec = captured[f"layer_{idx:02d}_output"][0, -1, :].float().cpu().contiguous()
        tensors[f"layer_{idx:02d}_output"] = vec

    tensors["post_final_norm"] = captured["post_final_norm"][0, -1, :].float().cpu().contiguous()

    last_logits = out.logits[0, -1, :].float().cpu()
    tensors["logits_full"] = last_logits.contiguous()

    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, 10)
    top10 = []
    for prob, idx in zip(topk.values.tolist(), topk.indices.tolist(), strict=True):
        top10.append({
            "token_id": idx,
            "token_str": tokenizer.decode([idx]),
            "prob": round(prob, 6),
            "logit": round(float(last_logits[idx]), 6),
        })

    ri_prob = float(probs[ri_id])
    ru_prob = float(probs[ru_id])

    suffix = f"_{tag}" if tag else ""
    out_bin = REPORTS_DIR / f"phase4zc_hf_hidden_states{suffix}.safetensors"
    save_file(tensors, str(out_bin))

    meta = {
        "attn_implementation": attn_impl,
        "dtype": dtype_name,
        "model_path": MODEL_PATH,
        "quantization": "none (full precision load, no bitsandbytes)",
        "forced_prefix": FORCED_PREFIX,
        "prompt_text_char_length": len(prompt_text),
        "full_text_char_length": len(full_text),
        "n_tokens": n_tokens,
        "n_hidden_state_entries": n_layers_plus_embd,
        "tensor_names_saved": list(tensors.keys()),
        "ri_token_id": ri_id,
        "ru_token_id": ru_id,
        "ri_prob": round(ri_prob, 6),
        "ru_prob": round(ru_prob, 6),
        "ri_minus_ru_prob": round(ri_prob - ru_prob, 6),
        "ri_rank": int((probs > probs[ri_id]).sum().item()) + 1,
        "ru_rank": int((probs > probs[ru_id]).sum().item()) + 1,
        "top10": top10,
        "elapsed_sec": round(time.time() - t0, 2),
        "output_file": str(out_bin),
    }
    out_meta = REPORTS_DIR / f"phase4zc_hf_hidden_states_meta{suffix}.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved -> {out_bin}")
    print(f"Saved -> {out_meta}")
    print(f"ri_prob={ri_prob:.6f} ru_prob={ru_prob:.6f} diff={ri_prob - ru_prob:.6f}")
    for r in top10:
        print(r)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--device-map", default="cuda:0")
    args = parser.parse_args()
    sys.exit(main(attn_impl=args.attn_impl, dtype_name=args.dtype, tag=args.tag,
                   device_map=args.device_map))
