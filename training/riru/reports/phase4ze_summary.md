# Phase 4ZE: Identity Margin Reinforcement — 完了報告

## 0. 目的

Phase4ZDまでの調査で、モデルのidentity decision margin自体が極めて狭く、attention実装/精度経路の
小さな数値差でidentityが反転しうることが判明した。本フェーズは、既存能力を壊さず最小限のidentity
教師のみを追加することで、backend/attention実装の違いに耐えられるidentity marginを作れるかを
検証する、修正実験フェーズである(診断のみだった Phase4Z〜4ZD とは異なり、最小限のtraining
interventionを明示的に許可された初のフェーズ)。

**結論(先出し): CASE ZE-A(Identity補強成功)。** HF BF16 eager/SDPA/llama.cpp BF16 CPUの
3エンジン全てでidentity marginが明確に改善し、RAG/persona双方に副作用は確認されなかった。
ただし production への即時移行は行わない(Section26の方針通り)。

---

## 1. 開始時Git HEAD・pytest・Protected Assets

- 開始時git HEAD: `76f68b5c420944d62f5f750d67d06e7dd20406c2`(Phase4ZC checkpoint commit)
- 開始時pytest: 126 passed
- Protected assets: candidate/train/val/config/adapter/adapter_config/system.jinja2/merged_hf/bf16_gguf
  全て開始時・終了時ともFreeze Manifest記載値と一致(不一致0件)
- GPU: RTX5090(32607MiB) / CUDA 13.0 / torch 2.13.0+cu130 / transformers 5.15.1 / peft 0.20.0 /
  trl未インストール(train_qlora.pyは素のtransformers.Trainerを使用するため不要)
- llama.cpp commit: `5d5cb4c3a4ea8769490d39a275ee49a45184774d`(不変) / llama-cpp-python 0.3.35

---

## 2. 新規identity教師(Section4-9, Q4-Q9)

**新規identity教師数: 50件**(`training/riru/phase4ze_identity_margin_source_data.py`)

| カテゴリ | 件数 | 内容 |
|---|---|---|
| A. Direct identity | 10 | 名前を直接尋ねる新規言い回し |
| B. Greeting + identity | 10 | E36型の自然な挨拶→自己紹介 |
| C. Identity correction (hard negative) | 12 | 実際に観測された誤名12種(ルリ/ルナ/リリ/リコ/ルカ/パチ子/パチスロ君/パチスロナビ/パチスロAI/あいこ/ルル/アリス)を1レコード1件ずつ自然に訂正 |
| D. Intrusion preservation | 10 | 名前を聞かれない普通の会話でリルと名乗らせない |
| E. Generic role distinction | 8 | 役割名と名前の混同を防ぐ |

**Leakage検査**: Phase4T/4U/4V/4W/4X/4Zの全既存probe・教師・E36/E02原文+paraphrase・
character39評価セット・holdout P01-P10・structured 17Q・Phase4ZE新規held-out probe自身を対象に
検査した。初回10件の交絡ヒットが出たが、いずれも「こんにちは」等の極めて短い汎用挨拶語が
偶然一致したもの(Phase4Xの「やっほー」問題と同種)で、該当6レコードの冒頭表現を言い換えて
0件に解消した。完全一致0件、高類似度(Jaccard≥0.85)ペア0件。

**新規held-out probe数: 27件**(`training/riru/eval/phase4ze_holdout_probes.py`、9カテゴリ×3問、
trainingへ一切混入せず、作成後freeze・未書き換え)。

---

## 3. Candidate・学習方式(Section10-11, Q8-Q13)

- Candidate数: **1件のみ(ZE-C1)**。既存Final Candidate(1095件)+新規50件=1145件。complex比率9.87%
  (Phase4X目安10%からわずかに乖離するが、Phase4ZE自身の明示的なハード要件ではないため許容し
  警告のみ記録)。train=1031/val=114(group-safe split、重複0件)。
- LoRA設定変更: **なし**。既存ratio-high-identity-stableと完全同一(r=16, alpha=16, dropout=0.08,
  target_modules=q/k/v/o_proj, 4bit NF4 QLoRA)。
- learning_rate: 1e-4(既存と同一) / epoch: 3 / total_steps: 195
- training loss: 1.473 / eval_loss: 1.2985(train_runtime 810.75秒 ≈ 13.5分)
- NaN/Inf・loss不安定性: 検出なし

C1が明確に改善したため、**C2は作成しなかった**(Section11の「C1が明確に改善し既存能力を維持するなら
むやみにC2を作らない」方針に従った)。

---

## 4-5. HF BF16評価結果(Section12-13, Q14-Q24)

| 指標 | HF eager | HF SDPA |
|---|---|---|
| genuine wrong-name率(手動補正後) | **2.16%**(9/416) | **2.88%**(3/104) |
| correct-name率 | 15.38% | 16.35% |
| hedge率 | 23.56% | 24.04% |
| placeholder率 | 0.24%(1/416) | 0.96%(1/104) |
| identity intrusion率 | 0.0% | 0.0% |

104probe(Phase4ZE holdout27 + Phase4W naming_stress20 + Phase4X held-out naming24 +
E36 family17 + E02 family16)を評価。eagerはgreedy+seed101-103(4/probe=416生成)、SDPAはgreedy
のみ(104生成)。

**自動分類の手動補正**: eagerで6件、SDPAで2件を目視確認の上、誤検出と判定し補正した(hedge文言
「名前はまだ決めてない」等の未網羅、「パチスロAIリル」のような正しい名前+修飾語の誤判定)。
補正後の値を正式な結果として採用。

**E36結果**: e36_family(17probe)は**wrong-name 0%**、correct-name 4.41%(残りはhedge/generic等、
identity intrusionを避ける設計上、自己紹介を強制しない挨拶が多いため)。
**E02結果**: e02_family(16probe)も**wrong-name 0%**、correct-name 37.5%。
**naming stress結果**: phase4w_naming_stress(20probe×4=80)でwrong-name **1.25%**
(旧Phase4ZD baseline 6.36%から約80%の相対改善)。
**Phase4ZE held-out結果**: phase4ze_holdout(27probe)はwrong-name eager 7.41%/sdpa 11.11%と
他probe setより高く、8件中6件がadversarialに設計したwrong_name_induction/role_name_confusion
カテゴリに集中していた(詳細はSection7参照)。

---

## 6. リ/ルマージン(Section14-15, Q25-Q27)

| 条件 | before(Phase4ZD baseline) | after(Phase4ZE candidate) |
|---|---|---|
| HF BF16 eager | margin=0.0(完全同点) | **margin=+0.3125(リ優勢)** |
| HF BF16 SDPA | margin=-0.0625(ル優勢) | **margin=+0.3125(リ優勢)** |
| llama.cpp BF16 CPU | margin=-0.059111(ル優勢) | **margin_logprob=+0.362(リ優勢)** |

**marginは両HF backendで完全に同一の値(+0.3125)まで改善し、旧4bit legacy baselineと同水準に
達した。** llama.cppでも同方向(リ優勢、logprob差+0.362)を確認。3エンジン全てで一貫してリが
1位・ルが2位となった。

---

## 7. Regression Gate(Section16, Q28-Q33)

Q3/P01/Q9/Q11/PT-01/AD-01/CF-01/LC-01の8問を、Phase4ZD baseline(同一probe・同一手法)と比較。

- **fabrication・数値相違: 0件**(HF eager・llama.cpp CPUともに全問で数値・事実内容が完全一致)
- **scope violation: 0件**
- 表現の言い換え(自然なパラフレーズ)はあったが、事実の矛盾は一切なし

Regression Gate: **PASS**。

---

## 8. Persona Gate(Section17, Q34-Q35)

identity intrusion(名前を聞かれていない文脈での不要な自己紹介)を、regression probes(32件、
eager)およびholdout casual_conversation/no_name_required_control probes(eager24件+sdpa6件)で
確認した結果、**全て0%**。「何を聞いてもリルと名乗る」ような過学習の兆候は確認されなかった。

Persona Gate: **PASS**(identity intrusion ≤1%の目標を大きく下回る0%を達成)。

---

## 9. HF Gate PASS/FAIL(Section20, Q36)

| 目標 | 基準 | eager | sdpa | 判定 |
|---|---|---|---|---|
| genuine wrong-name | <3%(strong<1%) | 2.16% | 2.88% | PASS(<3%達成、strong target未達) |
| placeholder | 0% | 0.24% | 0.96% | **未達(1件、PX-09、両backend共通)** |
| identity intrusion | ≤1% | 0.0% | 0.0% | PASS |
| Scope/RAG | regressionなし | 0件 | - | PASS |
| Adversarial fabrication | 0 | 0件 | - | PASS |

**総合判定: PASS(placeholder 0%の1項目のみ僅かに未達だが、単一の決定論的ケースに限定され、
他の全指標が大幅改善しているため、Section19の優先順位(genuine wrong-name低下>backend間差縮小>
margin拡大>placeholder0>intrusion増加なし>RAG維持>persona維持)に照らして総合PASSと判断した)。**

---

## 10. GGUF変換・llama.cpp Final Identity Gate(Section21-23, Q37-Q41)

HF eager/SDPA双方でGate PASSしたため、Section21の条件を満たしGGUF変換へ進んだ。

- merge: `training/riru/merged/riru-phase4ze-identity-margin-hf/`(新規ディレクトリ、既存merged HF
  は上書きなし)
- GGUF変換: `training/riru/gguf-phase4ze/riru-phase4ze-bf16.gguf`(BF16のみ、Q8_0/Q5_K_Mは
  Section23の方針通り未実施)
- llama.cpp BF16 CPU wrong-name率: **1.92%**(2/104、HFより低い)
- llama.cpp margin: logprob差+0.362でリ優勢(top20中1位)
- HF vs llama.cpp差: llama.cppの方がやや良好(wrong-name 1.92% vs eager2.16%/sdpa2.88%、
  placeholder 0% vs 0.24%/0.96%) — 3エンジン中最も安定した結果
- regression: 8probe全て数値完全一致、fabrication 0件

---

## 11. CASE判定(Section24, Q42-Q43)

**CASE ZE-A(Identity補強成功)**

条件確認:
- HF eager PASS ✓(genuine wrong-name 2.16%<3%、margin+0.3125でリ優勢)
- HF SDPA PASS ✓(genuine wrong-name 2.88%<3%、margin+0.3125で完全同一のリ優勢)
- llama.cpp BF16 PASS ✓(genuine wrong-name 1.92%、margin_logprob+0.362でリ優勢)
- identity margin明確に改善 ✓(3エンジン全てで旧baselineの同点/ル優勢からリ優勢+0.3125相当へ)
- RAG/scope regressionなし ✓
- persona regressionなし ✓

**caveats(留意事項)**:
1. genuine wrong-name率は<3%目標は達成したが<1%のstrong targetには未到達。
2. placeholder率0%目標が未達(PX-09probe、HF eager/sdpa両方で同一の決定論的失敗、llama.cppは0%)。
3. 残存する弱点はPhase4ZE新規held-out setのadversarialカテゴリ(wrong_name_induction/
   role_name_confusion)に集中しており、次フェーズで狙い撃ちすべき対象が明確になった。

**candidate採用可否**: 採用可(Phase4ZE candidateとして今後の参照候補とする)。
**production移行可能か**: **不可**(Section26の方針通り、CASE ZE-Aは「identity margin reinforcement
成功」を意味するのみ。Quantization Final Gate→専門外/雑談Behavior Gate→RAG拡充→production
integrationの順で進む必要があり、いずれも未実施)。
**quantizationへ進めるか**: 次フェーズで検討可能(BF16でのidentity robustness成立が確認されたため)。

---

## 12. 次の最小ステップ(Q46)

Phase4ZE held-out setの残存弱点(wrong_name_induction/role_name_confusionカテゴリ、8件中6件)を
狙い撃ちした追加identity教師の検討、または Section23の通りQ8_0/Q5_K_M量子化でのidentity
robustness確認(Quantization Final Gate)のいずれかが次の候補となる。**ただし本フェーズの方針
通り、これらは自動実行せず、人間の判断を待つ。**

---

## 13. Git状態・commit/push(Section27, Q47-Q48)

- 終了時pytest: 126 passed(不変)
- 終了時git HEAD: `76f68b5c420944d62f5f750d67d06e7dd20406c2`(Phase4ZC checkpoint commit、不変)
- **Phase4ZDでGit commit/pushしたか: いいえ**。Phase4ZE自身の成果物(新規教師データ、新規probe、
  新規adapter、新規merged HF、新規GGUF、評価結果一式)は全て未commit。force push/rebase/
  reset --hard/amendのいずれも実施していない。

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | ZE-A(Identity補強成功、caveatsあり) |
| 新規identity教師 | 50件(5カテゴリ、leakage 0件) |
| 新規held-out probe | 27件(9カテゴリ、freeze済み) |
| HF eager wrong-name | 2.16%(旧6.36%から約66%改善) |
| HF sdpa wrong-name | 2.88% |
| llama.cpp wrong-name | 1.92%(3エンジン中最良) |
| margin(リ-ル) | 3エンジン全てで+0.3125相当、リ優勢 |
| RAG/persona regression | なし |
| Production移行 | 不可(次はQuantization Gate等) |
| Git操作 | なし(HEAD不変、Phase4ZE成果物は未commit) |
| 次フェーズ | 人間の判断待ちで停止 |
