"""Phase 4C: リル人格QLoRA 学習スクリプト (Qwen2.5-14B-Instruct + RTX 5090)。

【重要】このスクリプトは「準備」段階のものであり、実行してはいけない。
`main()` は既定で `--dry-run` 相当の検証のみを行い、実際の `trainer.train()` は
明示的に `--i-confirm-start-training` フラグを渡さない限り呼び出されない
(コード上のセーフガード。ユーザーの明示的な許可なしに学習が始まらないようにするため)。

【要件との対応】
- ローカルHF形式のQwen2.5-14B-Instructを使用 (GGUFは推論専用、学習には使わない)
- bitsandbytes 4bit NF4 (BitsAndBytesConfig)
- PEFT LoRA (q_proj/k_proj/v_proj/o_proj のみ、MLP層は対象外)
- tokenizer.apply_chat_template() を使用 (<|im_start|>等の手動文字列結合はしない)
- assistant-only loss (system/userトークンはlabel=-100でマスク)
- train/val ロード (training/riru/processed/riru_train_v1.jsonl, riru_val_v1.jsonl)
- 学習ログ・eval loss・checkpoint・最終adapter保存、再開可能なcheckpoint構成
- OOM時に原因が分かるログ (GPUメモリ状況を例外時に出力)
- 本学習(smoke_test=False)の完了/失敗をSlack Incoming Webhookへ通知
  (Webhook URLはソースコードに書かず .env の SLACK_WEBHOOK_URL から読む。
  通知の送受信に失敗しても学習結果・adapter保存には一切影響しない設計)

TRLのSFTTrainerではなく、素の `transformers.Trainer` + カスタムDatasetで実装している。
理由: TRLのcompletion-only loss API はバージョンごとに変わりやすく、
本環境でのTRLバージョンが未確定な現段階では、`transformers.Trainer` + 手動マスキングの方が
挙動を明示的に管理でき、再現性・デバッグ容易性の面で安全と判断した
(TRLがインストールされている場合、SFTTrainerへの差し替えは容易な設計にしている)。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - .venv-qlora には導入済みだが念のため保護
    load_dotenv = None  # type: ignore[assignment]

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parents[1]
CONFIG_PATH = TRAINING_ROOT / "configs" / "qlora_config.json"
ENV_PATH = PROJECT_ROOT / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("riru_qlora")

# .env から SLACK_WEBHOOK_URL 等を読み込む。python-dotenv 自体が無い/失敗しても
# 学習処理には影響させない (Webhook URLをソースコードに直書きしないための仕組み)。
if load_dotenv is not None:
    try:
        load_dotenv(ENV_PATH)
    except Exception:  # noqa: BLE001 - .env読み込み失敗で学習を止めない
        logger.warning(".env の読み込みに失敗しました (Slack通知が無効になる可能性があります)。")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Slack Incoming Webhook 通知
# ---------------------------------------------------------------------------
# 要件:
#   - Webhook URLはソースコードに直接書かず、.env の SLACK_WEBHOOK_URL から取得する。
#   - 通知の送信に失敗しても (URL未設定・ネットワークエラー・HTTPエラー等)、
#     学習結果やadapter保存には一切影響させない (例外を外へ伝播させない)。
#   - 通知対象は本学習 (smoke_test=False) のみ。動作確認用のsmoke testは通知しない。
SLACK_NOTIFY_TIMEOUT_SEC = 10
SLACK_ERROR_SUMMARY_MAX_LEN = 500


def send_slack_notification(message: str) -> bool:
    """Slack Incoming Webhookへメッセージを送信する。

    どのような理由であれ失敗しても例外を送出しない
    (呼び出し元の学習フロー・adapter保存を止めないため)。
    戻り値は送信成否の記録用であり、呼び出し元はこれを無視してよい。
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        logger.warning(
            "SLACK_WEBHOOK_URL が .env に設定されていないため、Slack通知をスキップします。"
        )
        return False
    try:
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=SLACK_NOTIFY_TIMEOUT_SEC) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 200:
                logger.warning("Slack通知のレスポンスが異常です (status=%s)。", status)
                return False
        logger.info("Slack通知を送信しました。")
        return True
    except Exception as exc:  # noqa: BLE001 - 通知失敗は学習フローに影響させない
        logger.warning("Slack通知の送信に失敗しました (学習結果には影響しません): %s", exc)
        return False


def notify_training_success(
    *,
    duration_sec: float,
    epoch: float | None,
    step: int | None,
    train_loss: float | None,
    eval_loss: float | None,
) -> None:
    """学習正常終了をSlackへ通知する (失敗しても呼び出し元に影響させない)。"""
    try:
        duration_min = duration_sec / 60
        lines = [
            "✅ リル QLoRA学習完了",
            f"所要時間: {duration_min:.2f}分 ({duration_sec:.1f}秒)",
            f"epoch: {epoch if epoch is not None else 'N/A'}",
            f"step数: {step if step is not None else 'N/A'}",
            f"最終train loss: {train_loss if train_loss is not None else 'N/A'}",
            f"eval loss: {eval_loss if eval_loss is not None else 'N/A'}",
        ]
        send_slack_notification("\n".join(lines))
    except Exception as exc:  # noqa: BLE001 - 通知は学習フローに影響させない
        logger.warning("Slack成功通知の作成/送信中に予期しないエラー: %s", exc)


def notify_training_failure(error_summary: str) -> None:
    """学習異常終了をSlackへ通知する (失敗しても呼び出し元に影響させない)。"""
    try:
        text = error_summary
        if len(text) > SLACK_ERROR_SUMMARY_MAX_LEN:
            text = text[:SLACK_ERROR_SUMMARY_MAX_LEN] + "...(truncated)"
        lines = [
            "🚨 リル QLoRA学習失敗",
            f"エラー概要: {text}",
        ]
        send_slack_notification("\n".join(lines))
    except Exception as exc:  # noqa: BLE001 - 通知は学習フローに影響させない
        logger.warning("Slack失敗通知の作成/送信中に予期しないエラー: %s", exc)


# ---------------------------------------------------------------------------
# 学習中の記録用ユーティリティ (VRAM監視 / 構造化ログ / NaN・Inf検出)
# ---------------------------------------------------------------------------


def _nvidia_smi_snapshot() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        mem_used, mem_total, util = (x.strip() for x in out.stdout.strip().split(","))
        return {
            "vram_used_mib": int(mem_used),
            "vram_total_mib": int(mem_total),
            "gpu_util_pct": int(util),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


class GpuVramSampler:
    """バックグラウンドでnvidia-smiを定期ポーリングし、期間中の最大VRAMを記録する。"""

    def __init__(self, interval_sec: float = 1.0) -> None:
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_vram_mib = 0
        self.samples = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = _nvidia_smi_snapshot()
            if "vram_used_mib" in snap:
                self.max_vram_mib = max(self.max_vram_mib, snap["vram_used_mib"])
                self.samples += 1
            self._stop.wait(self._interval)

    def __enter__(self) -> GpuVramSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


class NaNInfError(RuntimeError):
    """学習中にloss/grad_normがNaNまたはInfになった場合に送出する。"""


def build_logging_callback():
    """TrainerCallbackを継承したロガーを生成する (importをrun_training内に閉じるため関数化)。

    step/epoch/train loss/eval loss/lr/grad_norm/step時間を構造化して記録する。
    NaN/Infを検出した場合は即座に例外を送出して学習を停止させる
    (無理に継続・再試行はしない方針のため)。
    """
    import math

    from transformers import TrainerCallback

    class RiruLoggingCallback(TrainerCallback):
        def __init__(self) -> None:
            self.records: list[dict] = []
            self._last_log_time = time.perf_counter()

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
            self._last_log_time = time.perf_counter()

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ARG002
            if logs is None:
                return
            now = time.perf_counter()
            elapsed = now - self._last_log_time
            self._last_log_time = now

            entry = {
                "step": state.global_step,
                "epoch": logs.get("epoch", state.epoch),
                "elapsed_since_last_log_sec": round(elapsed, 3),
            }
            entry.update(logs)
            self.records.append(entry)

            for key in ("loss", "grad_norm", "eval_loss"):
                val = logs.get(key)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fval) or math.isinf(fval):
                    raise NaNInfError(
                        f"{key} が NaN/Inf になりました "
                        f"(step={state.global_step}, value={val})。"
                        f"設定変更しての再試行はせず、ここで停止します。"
                    )

    return RiruLoggingCallback()


# ---------------------------------------------------------------------------
# データセット (messages形式 -> tokenize + assistant-only label マスク)
# ---------------------------------------------------------------------------


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def load_messages_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_assistant_only_example(tokenizer, messages: list[dict], max_seq_length: int):
    """1レコード分のmessagesを、assistant区間のみlossが有効なlabelsへ変換する。

    tokenizer.apply_chat_template() を使い、会話を1ターンずつ増やしながら
    トークン列の差分を取ることで、各assistantターンの開始・終了位置を特定する。
    system/user/roleヘッダ等のトークンはすべて label=-100 (無視) にする。
    """
    # NOTE: このtransformersバージョンの apply_chat_template(tokenize=True) は
    # 生のlist[int]ではなく BatchEncoding を返す (実機確認済み)。.input_ids で取り出す。
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True
    )["input_ids"]
    labels = [-100] * len(full_ids)

    # 1ターンずつ増やして、assistant発話の直前までのトークン数と、
    # assistant発話を含めた直後までのトークン数を比較し、差分区間を「学習対象」とする。
    running_messages: list[dict] = []
    prev_len = 0
    for msg in messages:
        running_messages.append(msg)
        ids_so_far = tokenizer.apply_chat_template(
            running_messages, add_generation_prompt=False, tokenize=True
        )["input_ids"]
        cur_len = len(ids_so_far)
        if msg["role"] == "assistant":
            # prev_len .. cur_len の区間がこのassistantターン (テンプレートのヘッダ含む可能性あり)
            # ヘッダ部分 (例: "<|im_start|>assistant\n") を除外するため、
            # このターン単体の文字列をトークナイズして末尾側から一致させる。
            start = prev_len
            end = cur_len
            if 0 <= start < end <= len(full_ids):
                labels[start:end] = full_ids[start:end]
        prev_len = cur_len

    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    attention_mask = [1] * len(full_ids)
    return TokenizedExample(input_ids=full_ids, labels=labels, attention_mask=attention_mask)


class RiruMessagesDataset:
    """messages形式JSONLを読み込み、assistant-onlyでtokenizeするDataset。"""

    def __init__(self, records: list[dict], tokenizer, max_seq_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        ex = build_assistant_only_example(self.tokenizer, rec["messages"], self.max_seq_length)
        return {
            "input_ids": ex.input_ids,
            "labels": ex.labels,
            "attention_mask": ex.attention_mask,
        }


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list[dict]) -> dict:
        import torch

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# モデル構築 (4bit NF4 QLoRA)
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(config: dict):
    """HF形式Qwen2.5-14B-Instructを4bit量子化でロードし、LoRAアダプタを付与する。

    NOTE: この関数は実際にモデルをロードするため大きなメモリ/VRAMを消費する。
    `--i-confirm-start-training` 指定時のみ呼び出される。
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_model_path = Path(config["base_model"]["local_path"])
    if not base_model_path.is_dir():
        raise FileNotFoundError(
            f"HF形式のベースモデルが見つかりません: {base_model_path}\n"
            f"GGUF (推論専用) とは別に、Hugging Face形式の "
            f"{config['base_model']['hf_repo_id']} が必要です。"
            f"ダウンロードにはユーザーの明示的な許可が必要です。"
        )

    bnb_cfg = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = config["lora"]
    # Phase 4Q: module別に異なるrank/alphaを指定する場合のみ使用 (例: o_projだけ低rank)。
    # config側にキーが存在しない場合は従来通りNone (=全module共通のr/lora_alpha) となり、
    # v1〜v4/v5-qkvの挙動は完全に不変。
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        rank_pattern=lora_cfg.get("rank_pattern") or {},
        alpha_pattern=lora_cfg.get("alpha_pattern") or {},
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# 学習本体
# ---------------------------------------------------------------------------


def run_training(config: dict, smoke_test: bool = False, smoke_test_steps: int = 4) -> int:
    """実際の学習を実行する (Slack通知の送受信で本学習フローに影響を与えないための薄いラッパー)。

    本学習 (smoke_test=False) が例外で終了した場合のみ、Slackへ失敗通知を送る。
    通知自体の失敗は `notify_training_failure` 内で完全に握りつぶされるため、
    ここでの通知呼び出しが元の例外の送出を妨げることはない。
    smoke testは動作確認用のため通知対象外とする。
    """
    if smoke_test:
        return _run_training_impl(config, smoke_test=smoke_test, smoke_test_steps=smoke_test_steps)
    try:
        return _run_training_impl(config, smoke_test=smoke_test, smoke_test_steps=smoke_test_steps)
    except Exception as exc:
        notify_training_failure(f"{type(exc).__name__}: {exc}")
        raise


def _run_training_impl(config: dict, smoke_test: bool = False, smoke_test_steps: int = 4) -> int:
    """実際の学習を実行する。

    smoke_test=True の場合、ごく少数サンプル・ごく少数stepのみ実行する動作確認モード
    (本学習ではない)。出力先を本番用ディレクトリとは別にする。
    --i-confirm-start-training が指定された場合のみ本学習(smoke_test=False)が呼ばれる。
    """
    import torch
    from transformers import Trainer, TrainingArguments

    train_cfg = config["training"]
    data_cfg = config["data"]
    out_cfg = config["output"]

    project_root = TRAINING_ROOT.parents[1]
    train_path = project_root / data_cfg["train_path"]
    val_path = project_root / data_cfg["val_path"]

    if smoke_test:
        adapter_dir = project_root / "training" / "riru" / "lora-riru-qwen-SMOKETEST-out"
        checkpoint_dir = adapter_dir / "checkpoints"
        log_dir = adapter_dir / "logs"
    else:
        adapter_dir = project_root / out_cfg["adapter_dir"]
        checkpoint_dir = project_root / out_cfg["checkpoint_dir"]
        log_dir = project_root / out_cfg["log_dir"]

    for d in (adapter_dir, checkpoint_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model/tokenizer (4bit NF4 QLoRA)...")
    model, tokenizer = build_model_and_tokenizer(config)

    train_records = load_messages_jsonl(train_path)
    val_records = load_messages_jsonl(val_path)
    if smoke_test:
        # ごく少数サンプルに限定する (学習品質ではなく配線の動作確認が目的)
        train_records = train_records[: max(smoke_test_steps * 4, 8)]
        val_records = val_records[:4]
        logger.warning(
            "SMOKE TEST MODE: train=%d val=%d, max_steps=%d (本学習ではありません)",
            len(train_records),
            len(val_records),
            smoke_test_steps,
        )
    else:
        logger.info("train=%d val=%d", len(train_records), len(val_records))

    max_seq_length = train_cfg["max_seq_length"]
    train_dataset = RiruMessagesDataset(train_records, tokenizer, max_seq_length)
    val_dataset = RiruMessagesDataset(val_records, tokenizer, max_seq_length)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    per_device_train_batch_size = int(train_cfg.get("_resolved_micro_batch_size", 2))
    gradient_accumulation_steps = int(train_cfg.get("_resolved_grad_accum", 8))

    # NOTE: transformers 5.x の TrainingArguments には warmup_ratio が存在しない
    # (実装確認済み: training_args.py に warmup_ratio の記述なし、warmup_steps のみ)。
    # そのため設定ファイル上の warmup_ratio を総stepから warmup_steps に換算して渡す。
    if smoke_test:
        total_steps_estimate = smoke_test_steps
    else:
        steps_per_epoch = max(
            1, -(-len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps))
        )
        total_steps_estimate = steps_per_epoch * train_cfg["num_train_epochs"]
    warmup_steps = max(0, round(train_cfg["warmup_ratio"] * total_steps_estimate))
    logger.info(
        "warmup_ratio=%s -> warmup_steps=%d (total_steps_estimate=%d)",
        train_cfg["warmup_ratio"],
        warmup_steps,
        total_steps_estimate,
    )

    training_args_kwargs = dict(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=train_cfg["learning_rate"],
        optim=train_cfg["optim"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        warmup_steps=warmup_steps,
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        seed=train_cfg["seed"],
        logging_steps=1 if smoke_test else train_cfg["logging_steps"],
        report_to=["none"],
    )
    if smoke_test:
        training_args_kwargs.update(
            max_steps=smoke_test_steps,
            eval_strategy="steps",
            eval_steps=smoke_test_steps,
            save_strategy="steps",
            save_steps=smoke_test_steps,
            save_total_limit=1,
            load_best_model_at_end=False,
        )
    else:
        training_args_kwargs.update(
            num_train_epochs=train_cfg["num_train_epochs"],
            eval_strategy=train_cfg["eval_strategy"],
            eval_steps=train_cfg["eval_steps"],
            save_strategy=train_cfg["save_strategy"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            load_best_model_at_end=train_cfg["load_best_model_at_end"],
            metric_for_best_model=train_cfg["metric_for_best_model"],
        )

    training_args = TrainingArguments(**training_args_kwargs)

    logging_callback = build_logging_callback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[logging_callback],
    )

    # 既存checkpointがあれば再開する (再開可能なcheckpoint構成)
    resume_from = None
    if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("checkpoint-*")):
        resume_from = True
        logger.info("Existing checkpoints found under %s; will resume.", checkpoint_dir)

    vram_before = _nvidia_smi_snapshot()
    logger.info("VRAM before training: %s", vram_before)
    train_wall_start = time.perf_counter()

    try:
        with GpuVramSampler(interval_sec=1.0) as sampler:
            trainer.train(resume_from_checkpoint=resume_from)
    except NaNInfError as exc:
        logger.error("=== NaN/Inf detected, stopping without retry === %s", exc)
        (log_dir / "training_log.json").write_text(
            json.dumps(logging_callback.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    except torch.cuda.OutOfMemoryError:  # noqa: PERF203
        logger.error("=== CUDA OutOfMemoryError ===")
        logger.error("per_device_train_batch_size=%s", per_device_train_batch_size)
        logger.error("gradient_accumulation_steps=%s", gradient_accumulation_steps)
        logger.error("max_seq_length=%s", max_seq_length)
        try:
            logger.error("torch.cuda.memory_summary():\n%s", torch.cuda.memory_summary())
        except Exception:  # noqa: BLE001
            logger.error("(memory_summary unavailable)")
        logger.error(
            "対処案: per_device_train_batch_sizeを下げる / "
            "gradient_accumulation_stepsを上げて実効バッチを維持する / "
            "max_seq_lengthを下げる / target_modulesをさらに絞る"
        )
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error("=== RuntimeError (likely OOM) === %s", exc)
            try:
                logger.error("torch.cuda.memory_summary():\n%s", torch.cuda.memory_summary())
            except Exception:  # noqa: BLE001
                pass
        raise

    train_wall_sec = time.perf_counter() - train_wall_start
    vram_after_train = _nvidia_smi_snapshot()

    logger.info("Saving final adapter to %s", adapter_dir)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    eval_metrics = trainer.evaluate()
    (log_dir / "final_eval_metrics.json").write_text(
        json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 学習中の記録一式を保存 (step/epoch/loss/eval_loss/lr/grad_norm/step時間/VRAM/総時間)
    (log_dir / "training_log.json").write_text(
        json.dumps(logging_callback.records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    training_summary = {
        "total_wall_clock_sec": round(train_wall_sec, 2),
        "total_wall_clock_min": round(train_wall_sec / 60, 2),
        "vram_before_training_mib": vram_before.get("vram_used_mib"),
        "vram_after_training_mib": vram_after_train.get("vram_used_mib"),
        "vram_peak_during_training_mib": sampler.max_vram_mib,
        "vram_samples": sampler.samples,
        "final_eval_metrics": eval_metrics,
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "total_steps_estimate": total_steps_estimate,
        "warmup_steps": warmup_steps,
    }
    (log_dir / "training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Training summary: %s", training_summary)
    logger.info("Final eval metrics: %s", eval_metrics)

    if not smoke_test:
        final_train_loss = None
        for entry in reversed(logging_callback.records):
            if entry.get("loss") is not None:
                final_train_loss = entry["loss"]
                break
        notify_training_success(
            duration_sec=train_wall_sec,
            epoch=round(trainer.state.epoch, 4) if trainer.state.epoch is not None else None,
            step=trainer.state.global_step,
            train_loss=final_train_loss,
            eval_loss=eval_metrics.get("eval_loss"),
        )

    return 0


# ---------------------------------------------------------------------------
# 準備段階の検証 (dry-run): ここまでは実行してよい
# ---------------------------------------------------------------------------


def dry_run_checks(config: dict) -> int:
    """実際のモデルロード・学習を行わず、準備状態のみを検証する。"""
    project_root = TRAINING_ROOT.parents[1]
    ok = True

    base_model_path = Path(config["base_model"]["local_path"])
    if base_model_path.is_dir():
        logger.info("[OK] HF形式ベースモデルが見つかりました: %s", base_model_path)
    else:
        ok = False
        logger.warning(
            "[未取得] HF形式ベースモデルが見つかりません: %s (GGUFのみ存在、学習には使えません)",
            base_model_path,
        )

    for key in ("train_path", "val_path"):
        p = project_root / config["data"][key]
        if p.is_file():
            n = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
            logger.info("[OK] %s: %d件 (%s)", key, n, p)
        else:
            ok = False
            logger.warning("[未生成] %s が見つかりません: %s", key, p)

    for pkg in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            __import__(pkg)
            logger.info("[OK] パッケージ利用可能: %s", pkg)
        except ImportError:
            ok = False
            logger.warning("[未インストール] パッケージが見つかりません: %s", pkg)

    try:
        import torch

        if torch.cuda.is_available():
            logger.info(
                "[OK] torch CUDA利用可能: %s (device=%s)",
                torch.__version__,
                torch.cuda.get_device_name(0),
            )
        else:
            ok = False
            logger.warning(
                "[未対応] 現在の torch (%s) はCUDAを利用できません "
                "(CPUビルドの可能性)。QLoRA学習にはCUDA対応torchが必須です。",
                torch.__version__,
            )
    except ImportError:
        pass  # 上のループで既に警告済み

    status = "READY" if ok else "NOT READY (上記の未対応項目を解消してください)"
    logger.info("dry-run 結果: %s", status)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help=(
            "使用する設定ファイルのパス (既定: configs/qlora_config.json = v1)。"
            "Phase 4G以降、v2等の別データ・別出力先で学習する場合は "
            "configs/qlora_config_v2.json のような専用configを指定する。"
            "学習ロジック自体はconfigの値に関わらず共通。"
        ),
    )
    parser.add_argument(
        "--i-confirm-start-training",
        action="store_true",
        help=(
            "823件×指定epochの本学習を開始する場合のみ指定する安全フラグ。"
            "指定しない限り dry-run (準備状態の検証) のみを行う。"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "ごく少数サンプル・ごく少数stepだけQLoRAの配線 (forward/backward/"
            "optimizer step/assistant-only loss/checkpoint保存) を確認する動作確認モード。"
            "本学習ではない。出力は本番adapterディレクトリとは別の "
            "lora-riru-qwen-SMOKETEST-out/ に保存される。"
        ),
    )
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        default=4,
        help="smoke-testモードでのoptimizer step数 (既定4、目安3〜5)。",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    logger.info("使用する設定ファイル: %s", config_path)
    config = load_config(config_path)

    if args.smoke_test:
        logger.warning(
            "SMOKE TEST モードが指定されました。max_steps=%d のみ実行します (本学習ではない)。",
            args.smoke_test_steps,
        )
        return run_training(config, smoke_test=True, smoke_test_steps=args.smoke_test_steps)

    if not args.i_confirm_start_training:
        logger.info("Dry-run mode (学習は開始しません; --i-confirm-start-training は未指定)")
        return dry_run_checks(config)

    logger.warning("学習開始フラグが指定されました。実際にQLoRA学習を開始します。")
    return run_training(config)


if __name__ == "__main__":
    sys.exit(main())
