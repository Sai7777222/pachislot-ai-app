# Phase 4U: ratio-high + Persona Identity 最小補強対照実験 最終報告

## 0. 結論の要約

- **誤名乗りは劇的に改善した。** naming probe(22問×11生成=242)でgenuine wrong-name率は v4 27.3% / ratio-high 23.1% → **ratio-high-identity 4.1%**（目標<5%を達成）。正しく「リル」と名乗る率も0%→**19.8%**へ改善。E02拡張評価では誤名乗り**0%**（v4:35%、high:55%から完全解消）、E36拡張ではplaceholderが**0/20**（v4/highの10%から解消）。
- **Q3/Q9/Q11/P04(scope-correct)は完全に維持された。** Q3 sampled recall 100%（greedy/sampled/min/max全て100%）、Q9独自計算hallucination 0/5、Q11全4カテゴリ0/5、P04のPhase4T方式required_fact_recallは99.1%（22probe平均）。identity intrusion(不要な自己紹介)は**0/48件**——過学習の兆候はなし。
- **一方、P01に実質的な回帰が確認された（90%→60%）。** P04とは異なりP01「天井について教えて」は正当な網羅要求質問であり、評価器由来の見かけ上の低下ではない。P02にも軽微な低下（76%→68%）が見られた。これは今回追加したidentity教師(43件、多くが短い名乗り応答)が、一部の「〜について教えて」型の広域網羅質問に対する完全性をわずかに希釈した可能性を示す、正直に報告すべき副作用である。
- **総合判定: ケースAに近いが、重要な留保付き。** 誤名乗り改善は極めて明確で採用基準を満たすが、P01の回帰は無視できない規模であり、次段階へそのまま進める前にP01型の「広域網羅」問題への対処（identity教師とcomplex教師のバランス再検討等）を検討する価値がある。

## 1. 開始前確認

- git HEAD: `d104ae4a6bc117bd4c8875140ff83d1b4232a3b0`（一致）。Phase4T未追跡ファイルは全て保全されていることを確認。
- pytest: 126 passed
- v4 / ratio-high / ratio-mid / o8 / o4 / v5-qkv adapter SHA-256、ratio-high candidate/train/val SHA-256、v4 candidate/train/val SHA-256、system.jinja2 MD5: 全て開始前後で不変

## 2. identity問題の再分類 (A〜G)

Phase4T naming probe全484生成(v4/high各242)を、改良した規則ベース分類器で再分類した(`phase4u_reclassify_naming.py`)。Phase4Tのreview_required率をそのまま真のwrong-name率として扱わず、以下のA〜G分類で再集計した。

| 条件 | A.誤名乗り | B.名前拒否 | C.placeholder | D.generic role | E.正しい「リル」 | F.検出器FP | G.その他 |
|---|---|---|---|---|---|---|---|
| v4 | 27.3% | 2.5% | 0.0% | 25.6% | 0.0% | 0.0% | 44.6% |
| ratio-high | 23.1% | 7.4% | 0.0% | 31.4% | 0.4% | 0.0% | 37.6% |

「登録名は『パチスロ博士』」のような固定cueパターンでは拾えない架空の登録名claimも追加パターンで捕捉し、genuine wrong-name率を精緻化した。

## 3〜4. identity教師データ設計

`training/riru/phase4u_identity_source_data.py`にidentity教師43件(識別positive 35件＋intrusion防止8件)を作成。**設計時にPhase4T naming probeとの文面重複が7件見つかり、全て異なる言い回しへ修正した**（例:「自己紹介して」→「自分のことを話してみて」）——これはcontamination検査(item6)が実際に機能した重要な検証例である。誤名(リコ/リサ/アリス等)を負例として列挙する方式は採らず、正しいidentity mappingのみを教える方針を徹底した。placeholder("〜〜"等)を含む教師は0件。

## 5. complex教師比率

既存complex教師113件は一切削除せず、新規43件を追加。complex比率は11.0%→**10.56%**（目安10%以上を維持、希釈は最小限）。

## 6. train/val split

group-safe split(seed=42)、train 964件/val 106件、**train/val overlap = 0**。Phase4T naming probe/P04 probe・実Q3・structured17・character39との文面重複検査を実施し、修正後は**contamination 0件**を確認。

## 7〜9. 学習

LoRA設定はratio-highと完全同一(q/k/v/o_proj、r=16/alpha=16/dropout=0.08、rank_pattern/alpha_pattern不使用)。preflight全項目READY確認後、1 candidateのみ学習。

| 項目 | 値 |
|---|---|
| train/val件数 | 964 / 106 |
| 総step | 183 |
| 学習時間 | 764.6秒(12.75分) |
| train_loss | 1.534 |
| 最終eval_loss | 1.096 |
| peak VRAM | 24052 MiB |
| NaN/Inf/OOM/CUDA error | なし |
| adapterサイズ | 100.7MB(v4/ratio-highと同一、LoRA構造一致を裏付け) |

## 10. Identity評価 (naming probe 22問)

| 指標 | v4 | ratio-high | **ratio-high-identity** |
|---|---|---|---|
| genuine wrong-name率 | 27.3% | 23.1% | **4.1%** |
| 正しく「リル」率 | 0.0% | 0.4% | **19.8%** |
| 名前拒否率 | 2.5% | 7.4% | 0.4% |
| generic-role-only率 | 25.6% | 31.4% | 20.7% |
| placeholder率 | 0.0% | 0.0% | 0.0% |

review_requiredとgenuine wrong-nameを分離した上で、全wrong-name候補を目視確認済み。**目標(<5%)を達成し、理想(0%)に近い水準まで改善した。**

## 11. E36/E02重点評価 (各20seed)

| 指標 | E36 v4 | E36 high | **E36 identity** | E02 v4 | E02 high | **E02 identity** |
|---|---|---|---|---|---|---|
| genuine wrong-name率 | 25.0% | 10.0% | **5.0%** | 35.0% | 55.0% | **0.0%** |
| placeholder率 | 10.0% | 10.0% | **0.0%** | 0.0% | 0.0% | 0.0% |
| 正しい「リル」率 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **25.0%** |

E02(ratio-highが55%と最も悪化していた設問)で誤名乗りが完全に解消された点が特に顕著。E36のplaceholder(「私は〜〜だよ」)も完全解消。

## 12. 回帰評価

| 指標 | ratio-high(Phase4S) | **ratio-high-identity** |
|---|---|---|
| Q3 sampled avg recall | 100.0% | **100.0%**（維持） |
| Q3 greedy recall | 100.0% | **100.0%**（維持） |
| P01(旧context-all-fact) | 90.0% | **60.0%（回帰）** |
| P02(旧context-all-fact) | 76.0% | 68.0%（軽微な低下） |
| P04(旧context-all-fact) | 33.3% | 33.3%（不変、既知の評価器限界） |
| P04(Phase4T方式required_fact) | 98.2%(22probe平均) | **99.1%**（維持・微増） |
| Q9独自計算hallucination | 0/5 | 0/5（維持） |
| Q11全4カテゴリ | 全て0/5 | 全て0/5（維持） |
| structured17平均長 | 57.4字 | 57.2字（維持） |
| character39平均長 | 約31.2字 | 30.4字（維持） |
| identity intrusion | - | **0/48（発生なし）** |

**P01の目視確認**: 「天井について教えて」という広域網羅型の質問(P04の「差だけ」とは異なりQ3同様の完全列挙が求められる設問)で、5seed中4seedがゲーム数(450G/750G/1300G)のみを回答し、確率(18%/27%/55%)を省略していた(1seedのみ完全回答)。これはPhase4Rで確立した「評価器の見かけ上の低下」パターンとは異なり、**真の情報省略の再発**である。identity教師43件の追加が、一部の「〜について教えて」型設問に対する完全性をわずかに希釈した可能性が高い。P02でも同様の傾向が軽微に見られた(1/5seedで完全な0%回答)。

## 13〜14. 採用基準の評価

| 基準 | 結果 |
|---|---|
| genuine wrong-name <5%(目標) | ○ 4.1% |
| genuine wrong-name 0%(理想) | △ 未達だが大幅改善 |
| placeholder <= ratio-high | ○ (E36で0%、ratio-highの10%以下) |
| 正しい「リル」率の明確な改善 | ○ 0.4%→19.8% |
| Q3 sampled recall >=90% | ○ 100% |
| 3ゲーム数完全/3%完全 | ○ (全seed達成) |
| Q9 hallucination=0/5 | ○ |
| Q11ヤメ時=0/5 / causal=0/5 | ○ |
| Phase4T P04型 required_fact_recall>=95% | ○ 99.1% |
| character39で重大な新規回帰なし | ○ (長さ・傾向とも維持) |
| 全回答自己紹介化(identity overfitting)なし | ○ 0/48 |
| **P01/P02の維持**(item12で明示的に要求) | **× P01が90%→60%で回帰** |

**identity/RAG/Scope/Personaの主要基準はほぼ全て満たしているが、item12で明示的に再評価を求められたP01に無視できない回帰が確認された。**

## 15. 結果別判断

Phase4U指示書のケースA〜Dのうち、**ケースAの大部分を満たすが、B寄りの留保が必要**な結果である:

- wrong-nameは大幅改善（ケースAの前半に合致）。
- Q3/Q9/Q11/P04(scope-correct)は維持（ケースAの条件に合致）。
- しかし**P01という「広域網羅」型の設問で真の回帰が発生**しており、これは「RAGが回帰」という点でケースBの要素を含む。

**提案**: ratio-high-identityは誤名乗り改善において非常に強い成果を示しており、次段階候補として有望である。ただし正式候補化の前に、(a) P01型「〜について教えて」広域質問に対する完全性の再検証(より多くのseed・より多様なheld-outでの確認)、(b) 必要であればidentity教師の件数・比率の微調整(今回43件・complex比率10.56%)、を追加で検討することを推奨する。Phase4U内では追加学習は一切行っていない。

## 16. 禁止事項の遵守

Phase4U内で追加sweep・複数candidate学習・ratio変更・LoRA構造変更・system prompt変更・merge・GGUF・正式採用は一切行っていない。1 candidateのみ学習した。

## 17. 最終確認

- pytest: **126 passed**
- protected assets: v1〜ratio-high adapter SHA-256、ratio-high candidate/train/val、v4 candidate/train/val SHA-256、system.jinja2 MD5 全て不変
- git diff: 既存追跡ファイルへの差分なし
- git status: 新規作成ファイルのみ(`??`)、Git commit/pushは実施していない

## 作成ファイル一覧

- `training/riru/eval/phase4u_reclassify_naming.py` / `training/riru/reports/phase4u_naming_reclassification.json`
- `training/riru/phase4u_identity_source_data.py`
- `training/riru/build_phase4u_dataset.py` / `reports/phase4u_dataset_quality.json` / `phase4u_dataset_summary.json`
- `training/riru/processed/riru_ratio_high_identity_candidate.jsonl` / `_train.jsonl` / `_val.jsonl`
- `training/riru/configs/qlora_config_ratio_high_identity.json`
- `training/riru/phase4u_pretrain_checks.py` / `reports/phase4u_pretrain_checks.json`
- `training/riru/lora-riru-qwen-ratio-high-identity/`（adapter・checkpoint一式）
- `training/riru/eval/phase4u_comprehensive_eval.py` / `phase4u_comprehensive_results.json`
- `training/riru/eval/phase4u_analyze.py` / `reports/phase4u_evaluation_analysis.json`
- `training/riru/reports/_phase4u_contamination_utf8.txt` / `_phase4u_contamination2_utf8.txt` / `_phase4u_p01p02_spotcheck_utf8.txt`
- `training/riru/reports/phase4u_summary.md`（本ファイル）

## 停止

学習・評価・目視確認・pytest・保護対象資産確認・レポート作成が完了しました。merge/GGUF化・正式採用・追加学習・Phase 4V等への自動移行・Git commit/pushは一切行っていません。次のご判断をお待ちします。
