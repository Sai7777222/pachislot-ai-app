# Phase 4R: 教師データ Fact Retention / Information Selection 監査 最終報告

## 0. 結論の要約

- **自動評価器の目視検証（32件）により、重大なfalse negativeバグを発見・特定した**: 非数値(categorical)factを「原文節の完全一致」で判定する箇所が、自然な言い換え（語順入替・助詞変化等）を一切許容せず、LOW-retention判定された10件を目視確認したところ**10/10 (100%) が偽陰性**だった。一方、数値・%・mapping系factを含む22件は**22/22 (100%) が真陽性**だった。
- この検証結果に基づき、**最終結論の主根拠には`numeric_normalized`/`percentage_normalized`/`mapping_normalized`（信頼できる）を用い、`overall_normalized`（categorical facts混入により汚染されている）は主根拠に使わない**。
- **信頼できる指標で見る限り、教師データのfact retentionは極めて高い**（percentage_normalized=100%、mapping_normalized=100%、Q3-like subset=96.7%、T1-0=100%）。**Hypothesis D（教師データ全体でretentionが低い）は明確に反証された。**
- **最も強く支持されるのは、Hypothesis Bの精緻化版**: 教師データ自体の「質」は高いが、Q3のような複数fact・percentage・mappingを要する複雑な入力に正しく対応する教師例（relevant facts≥5 かつ percentage・mappingあり = 「Q3-like」）は**914件中わずか6件（0.66%）**しか存在しない。さらに、v4のQ3実際の圧縮回答「天井ゲーム数は3種類あって、抽選で決定するよ。」に文体的に最も近い教師例は、`faithful_to_given_info`カテゴリの**「特化ゾーンは3種類あるみたいだよ。」**（件数を述べて終わる、内訳を列挙しない回答）であり、この「件数を言って止める」構造が教師データ内に実在することを確認した。
- **Phase4K追加17件の位置づけ**: 信頼できる指標（percentage/mapping）では既存897件・追加17件ともに100%で差はない。以前観測された「Phase4K17件の方がretentionが低い」という結果はcategorical false negativeバグによる測定誤差であり、実質的な品質差ではないと判断する。
- **次フェーズへの提案根拠は得られた**: ただし「データ件数を増やす」ではなく、「Q3型の複雑な複数fact入力を完全に列挙する教師例の“比率”を、既存の短文・単一fact回答に対して意味のある密度まで引き上げる」方向を提案する（件数の絶対的な追加ではなく、構造・比率の是正）。

## 1. 保護対象資産の開始時/終了時確認

Phase4Q終了時点を基準として記録し、Phase4R終了時に再確認。全て不変。

| 項目 | 値 |
|---|---|
| git HEAD | `2e0492d`（不変） |
| git status | `train_qlora.py`のみ`M`（Phase4Q由来、Phase4Rでは無変更） |
| v1/v2/v3/v4/v5-qkv/o8/o4 adapter SHA-256 | 全てPhase4Q終了時点と一致 |
| riru_train_v4.jsonl SHA-256 | `d331fef4...`（不変） |
| riru_val_v4.jsonl SHA-256 | `3df5c2a8...`（不変） |
| riru_lora_v4_candidate.jsonl SHA-256 | `341c44d0...`（不変） |
| system.jinja2 MD5 | `f3ea72a9...`（不変） |
| train/val件数 | 823 / 91（不変） |

## 2〜3. 監査対象・データスキーマ（実データから確認）

train 823件・val 91件・candidate 914件（train+val）を対象とした。実データを読み、以下のスキーマを確認:

```
{"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}, ...],
 "metadata": {"source": ..., "category": ..., "category_code": ..., "index"|"legacy_index": ..., ...}}
```

**system roleは存在しない**。RAG的contextはuser contentの中に直接埋め込まれている。2つの主要形式を確認:
1. 簡易形式: `"参照情報：A、B、C\n<question>"`
2. Phase4K構造化形式: `"【対象機種】...\n【構造化データ】\n- [label] key: value\n...\n【関連する解説文章】\n◆ 見出し（出典カテゴリ: xxx）\n本文\n...\n<question>"`

metadata.source別内訳: `phase4b_generated`(300) / `phase4f_generated`(74) / `phase4k_generated`(17) / それ以外`legacy`(523、`source_file`にlegacy jsonl名が入る)。category_code A〜Kおよびpersona系(legacy)、T1(complex_rag_structure_omission_prevention,11)・T2(derived_entity_retention,6)がPhase4K追加分。

## 4〜10. Fact抽出・relevance判定・retention計算の設計

- deterministic抽出: 構造化行`- [label] key: value`からmapping fact、`◆見出し（出典カテゴリ）\n本文`から埋め込み数値、簡易`参照情報：`形式から節単位でPhase4Jの既存ロジック(bug修正済み)を踏襲して数値/定性factを抽出。
- 正規化: `％→%`、カンマ除去、`ゲーム→G`統一。exact/normalized双方を分離して保持。
- relevance判定: Level1(ラベル・質問双方のtopic語集合の一致)+Level2(片方のみtopic語がある場合の緩やい部分一致)、confidence(high/medium/low)を保持。high-confidence subsetは別途集計。
- category別集計: numeric / percentage / mapping / condition / exceptionを分離して算出。

## 11〜30. 主要な統計結果（item32の23問への回答）

**1. 教師データ全体のfact retentionは何%か** — `overall_normalized`（categorical facts含む生の値）は mean 75.4% / median 100.0%だが、item28の目視検証によりこの指標にはfalse negativeバグが混入していることが判明したため、単独では信頼しない。

**2. numeric retentionは何%か** — 対象105件中のnumeric系facts、目視検証と整合的に高水準（`phase4r_fact_retention_results.json`の`numeric_normalized`参照、percentage/mappingと同様100%近傍）。

**3. percentage retentionは何%か** — **100.0%**（該当26件全て、min=max=100.0%）。

**4. mapping retentionは何%か** — **100.0%**（該当9件全て、min=max=100.0%）。

**5. Q3-like subsetは何件あるか** — relevant facts≥5 かつ percentage かつ mappingを満たす記録は**6件**（914件中0.66%）。ID: train#237(T1-0)/489/636/669/683/763。

**6. Q3-likeの平均retentionは何%か** — overall 96.7%、percentage 100.0%、mapping 100.0%、平均回答長73.2字。

**7. fact数が増えるほどretentionは落ちるか** — **落ちない。むしろ逆**: 1fact=70.8%→2facts=76.5%→3-4facts=91.7%→5-7facts=97.5%と、fact数が多い記録ほど自動判定のretentionが高い。ただしこれは「1factの多くがcategorical false negativeの影響を強く受ける」ことの裏返しでもあり(1fact記録はcategorical単発が多い)、数値のみで見ればfact数増加によるretention低下という当初仮説は支持されなかった。

**8. answer lengthとretentionに関係はあるか** — Pearson r=0.213（弱い正の相関、負ではない）。「短い回答ほどfactを捨てている」という単純な関係は、少なくともこの弱い相関係数からは支持されない。

**9. compression ratioとretentionに関係はあるか** — Pearson r=0.068（ほぼ無相関）。

**10. train/valで差があるか** — `by_source`/train-val集計を参照。両者とも同様の傾向を示し、val側だけが極端に短い・percentage retentionが低いといった偏りは確認されなかった。

**11. Phase4K 17件は既存データより高retentionか** — 信頼できる指標（percentage/mapping）では既存897件・追加17件ともに**100%で差なし**。overall_normalized単体では17件の方が低く出たが、これはcategorical false negativeバグの影響であり実質的な差ではないと判断する。

**12. T1-0自体は適切な教師例か** — **極めて適切**。7件のrelevant facts（天井3種のゲーム数×%マッピング3組＋到達処理＋追加抽選＋成功率）全てを100%保持し、簡潔さも保ちながら省略ゼロを達成している模範例。

**13. Q3の短縮回答に近い教師パターンは実際に存在するか** — **存在する**。char-bigram類似度でQ3代表回答「天井ゲーム数は3種類あって、抽選で決定するよ。」に最も近いのは`faithful_to_given_info`カテゴリの「特化ゾーンは3種類あるみたいだよ。」（sim=0.152）——「件数を言って止める」という同一の文構造を持つ。

**14. そのパターンは何件程度存在するか** — 完全一致する構造パターン（`は◯種類あって/あり`+`抽選で決定する/決まる`の組み合わせ）自体は914件中**0件**（Q3の具体的な言い回しの直接的コピー元は存在しない）。しかし「件数のみ述べて内訳を列挙せず終わる」という短い定性的スタイル自体は`faithful_to_given_info`(50件)を中心に複数存在することを確認した。

**15. percentage/mappingが特に捨てられやすいか** — **捨てられていない**（100%）。当初想定と反対の結果。

**16. persona subsetが短文化傾向を強めている証拠はあるか** — 限定的。fact-bearingカテゴリ(109件, 11.9%)の平均回答長34.9字 vs persona様カテゴリ(805件, 88.1%)の平均回答長31.1字と、**差は小さい**。むしろ「RAGラベルの付いたカテゴリの大多数(faithful_to_given_infoの50件等)自体が単一fact・短文回答であり、persona/RAGという二分法よりも『単純1fact-QA(多数)』対『複雑多fact-QA(極少数=T1系11件)』という軸の方が、観測された偏りをよく説明する」というのが本フェーズの精緻化された結論である。

**17. 自動fact evaluatorの目視精度は十分か** — **overall_normalizedは不十分**（categorical facts判定にfalse negative）。**percentage/mapping/numeric_normalizedは十分**（目視22/22で真陽性、false positiveなし）。

**18. Phase4Rの結果はHypothesis A〜Eのどれを支持するか** — 下記「25. 仮説判定」参照。

**19. Q3問題の主要因として「教師データの情報選択方針」を疑う根拠は強まったか** — **形を変えて強まった**。個々の教師回答が「情報を捨てる」ことを教えているという単純な仮説は反証されたが、「複雑な複数fact入力に完全に応答する教師例が、絶対数・相対比率ともに極端に少ない」という、より精緻化された形の「情報選択方針の教師データ設計問題」は強く支持される。

**20. 次に新規学習を行う科学的根拠が得られたか** — 部分的に得られた。ただしPhase4N〜4QのLoRA構造探索とは異なる角度（教師データの構造比率）からのアプローチが必要という結論。

**21. 得られた場合、変更すべきなのは件数・教師構造・fact retention・別要因のどれか** — **件数の単純追加ではなく「教師データの構造比率」**。既存のfact-bearing教師例(特にT1的な複雑構造)自体の retention・品質は既に高いため、量を増やすことよりも、単純1fact-QA例に対する複雑多fact-QA例の**相対頻度**を引き上げることが本筋と考えられる。

**22. v4/o8/o4/v5-qkvの追加探索を続ける意味があるか** — Phase4N〜4Qの結論（LoRA適用強度・構造調整だけでは三軸を両立できない）は本フェーズの結果と矛盾しない。教師データ側の構造比率という新しい変数を制御しないまま、LoRA構造だけをさらに調整する追加探索は、優先度が低いと判断する。

**23. merge/GGUFへ進める状態か** — いいえ。本フェーズは診断のみであり、v6等は作成していない。

## 悪い例・良い例の目視確認

`_phase4r_worst_cases_utf8.txt`（overall/percentage/mapping/high-density各worst30）、`_phase4r_good_cases_utf8.txt`（good cases 7件＋T1-0詳細）を作成し全件目視した。worst caseの大多数はfalse negativeであり、真の省略と確認できたケースはごく少数に留まった（詳細は当該ファイル参照）。

## Phase4K 17件個別識別

`metadata.source == "phase4k_generated"`で17件（T1×11, T2×6）を機械的に識別し、既存897件と分離して比較した（11節参照）。

## 教師データ内の「圧縮パターン」探索

- `だよ。`終端: 100件(10.9%)、`なんだ。`: 17件(1.9%)、`だね。`: 9件(1.0%)
- 「Xは◯種類あって/あり」+「抽選で決定する/決まる」の完全構造一致: **0件**
- Q3出力への最近傍（char-bigram類似度）: 上位に`faithful_to_given_info`の短い定性回答と、`complex_rag_structure_omission_prevention`(T1)のフル回答が混在——文体的scaffolding（「は◯種類あって」）は共有しつつ、後者は完全列挙・前者は列挙なしで終了、という対照が明確に観察された。

## Persona subset vs RAG-labeled subset

fact-bearingカテゴリ(109件): 平均34.9字。それ以外(805件): 平均31.1字。差は小さく、「personaデータが全体を短文化させている」という仮説は強くは支持されない（16節参照）。

## 自動評価器の検証（item28）

32件の層化ランダムサンプルを目視。LOW群10/10がfalse negative、HIGH/PCT/MAP群22/22がtrue positive。詳細は`_phase4r_manual_validation_utf8.txt`。

## wrong-name detectorの改善提案（item29、実装は行わず提言のみ）

Phase4Qで発見された「あいだっち」のように、固定リスト方式の誤名検出は新規パターンを取りこぼす。改善案（本フェーズでは実装しない）:
1. E36等の自己紹介系プロンプトの回答から、固定文「私は」「僕は」直後の名詞句を正規表現で抽出し、既知の正しい名前候補（「リル」）と厳密一致しない場合を全て「要目視確認」としてフラグする方式への切り替え。
2. 完全自動判定はfalse negativeのリスクが残るため、目視確認を必須とする運用を維持する。

## 25. 仮説判定

- **Hypothesis A（教師データ全体が高retention）**: 信頼できる指標（percentage/mapping/numeric）で見る限り**支持される**。
- **Hypothesis B（全体は高いがQ3-like/high-densityだけ低い）**: 「低い」という部分は反証されたが（Q3-likeもむしろ96.7%と高い）、**「Q3-likeな教師例の絶対数・相対比率が極端に少ない」という精緻化された形で強く支持される**。
- **Hypothesis C（percentage/mappingだけ特異的に低い）**: **明確に反証**（100%）。
- **Hypothesis D（教師データ全体でretentionが低い→LoRAが省略方針を学習した）**: **反証**。個々の教師回答自体は省略を教えていない。
- **Hypothesis E（Phase4K17件は高retentionだが既存897件と分布が大きく異なる→17件では上書きできなかった）**: **支持される**。retention自体は同水準だが、17件（T1は11件のみ）という絶対数が、823件の学習セット全体（うち大多数が単一fact・短文回答）に対してあまりに少数であり、分布を動かすには至らなかった可能性が高い。

**総合**: A（個々の教師品質は高い）+ B/E（複雑構造教師例の比率的少数性）が最も強く支持される組み合わせであり、C/Dは反証された。

## 26. 因果の断定について

本フェーズは観察研究であり、上記はいずれも「強く支持する」「整合的である」「主要候補である」という表現に留め、「これが100%の原因」という断定はしていない。

## 28. pytest

**126 passed**（不変）。

## 作成ファイル一覧

- `training/riru/audit/phase4r_fact_retention_audit.py`
- `training/riru/audit/phase4r_density_qlike_analysis.py`
- `training/riru/audit/phase4r_teacher_pattern_analysis.py`
- `training/riru/reports/phase4r_fact_retention_results.json`
- `training/riru/reports/_phase4r_full_records.json`
- `training/riru/reports/phase4r_fact_density_analysis.json`
- `training/riru/reports/phase4r_q3like_analysis.json`
- `training/riru/reports/phase4r_density_qlike_combined.json`
- `training/riru/reports/phase4r_teacher_pattern_analysis.json`
- `training/riru/reports/_phase4r_worst_cases_utf8.txt`
- `training/riru/reports/_phase4r_good_cases_utf8.txt`
- `training/riru/reports/_phase4r_manual_validation_raw_utf8.txt`
- `training/riru/reports/_phase4r_manual_validation_utf8.txt`
- `training/riru/reports/_phase4r_q3_nearest_utf8.txt`
- `training/riru/reports/_phase4r_schema_explore_utf8.txt` / `_phase4r_schema_explore2_utf8.txt`
- `training/riru/reports/phase4r_summary.md`（本ファイル）

## 停止

解析・目視検証・pytest・protected asset確認・最終レポート作成が完了しました。新規学習・v6 candidate作成・merge・GGUF化・Git操作は一切行っていません。次のご判断をお待ちします。
