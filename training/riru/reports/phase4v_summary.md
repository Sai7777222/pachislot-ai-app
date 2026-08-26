# Phase 4V: Broad-Question Completeness 回帰診断 最終報告

## 0. 結論の要約

- **P01の回帰(90%→60%)は held-out probeでは一般化しなかった。** 完全に架空の6 context family×6質問文体=36問(864生成)で評価した結果、ratio-highとratio-high-identityのrequired_fact_recallは**96.7% vs 96.9%（ほぼ同一、identityがわずかに上回る）**、complete_answer_rateも**85.6% vs 87.2%（identityが上回る）**だった。
- **paired比較（同一probe・同一seed）は180ペア中169件が完全同点(tie)、7件でidentityが優位、identityが劣るのはわずか4件(2.2%)のみ**で、平均差は+0.2pt（identity優位）。
- **4件のloss全てを目視確認した結果、いずれも真の壊滅的省略ではなかった。** 2件は副次的な単一categorical fact(「RT確定」)のみの欠落（数値・%は完全保持）、残り2件は「550G〜1350Gまで」という自然な範囲表現や「出目Pの示唆はL」という自然な語順変化による評価器側の偽陰性（Phase4R/4Tで確立済みの既知の限界パターン）だった。
- **broad/narrow別でも有意な乖離はなかった**(broad: high 95.1% vs identity 95.4%、narrow: 両者とも100.0%)。fact種別(percentage/numeric/categorical)別でもidentityがhighを下回る項目はなかった。
- **判定: CASE B。** P01の回帰は局所的/seed依存の事象であり、identity教師追加による一般的な broad-question completeness の低下ではない。ratio-high-identityを次段階の最終候補として、より広い最終評価へ進める根拠が得られた。

## 1. 開始前確認

- git HEAD: `d104ae4a6bc117bd4c8875140ff83d1b4232a3b0`（一致）。Phase4T/4U成果物は全て保全。
- pytest: 126 passed
- v4 / ratio-high / ratio-high-identity adapter SHA-256、各candidate/train/val SHA-256、system.jinja2 MD5: 開始前後で不変

## 2〜3. 新規held-out probe設計 (36問)

`training/riru/eval/phase4v_probes.py`にて、P01/P02/Q3の文面・数値・エンティティを一切使わない6つの架空context family(天井/設定別確率/複数モード/ゾーン/示唆対応/例外条件分岐)を作成し、各familyに6種類の質問文体(broad_topic/overview/explain/tell_me_all/specific_complete/narrow_control)を適用して36問を構成した。各probeについてrequired_facts/optional_facts/irrelevant_factsを明示的に分離し、**broad質問ではtopic内の主要fact全てをrequired、narrow質問では質問対象のfactのみをrequiredとする**設計を徹底し、P04で判明した「context中の全factを必須扱いする」評価ミスを再発させないようにした。

## 4. contamination検査

Phase4I P01/P02/P04、実Q3(構造化データ含む)、Phase4T naming/P04-type probe、Phase4U identity教師、structured17、character39、ratio-high complex教師との文面重複を検査した結果、**当初2件（「天井について教えて」「モードについて教えて」がP01/P07と完全一致）を検出し、言い換えて修正**（「天井の仕組みについて教えて」「モードの仕組みについて教えて」）。修正後は文面重複0件、実数値(510G/450G/18%等)重複も0件を確認。

## 5〜6. 評価条件

A_base/B_v4/C_ratio_high/D_ratio_high_identityの4条件、各36問についてgreedy+5seed(42-46)、計**864生成**を実施。学習は一切行っていない。

## 7. 主要結果

### 全体 (required_fact_recall / complete_answer_rate)

| 条件 | mean_recall | median | complete_answer_rate |
|---|---|---|---|
| A_base | 83.7% | 100% | 59.4% |
| B_v4 | 93.2% | 100% | 77.2% |
| C_ratio_high | 96.7% | 100% | 85.6% |
| **D_ratio_high_identity** | **96.9%** | 100% | **87.2%** |

### broad vs narrow

| 条件 | broad recall | narrow recall | broad complete率 | narrow complete率 |
|---|---|---|---|---|
| C_high | 95.1% | 100.0% | 78.3% | 100.0% |
| D_identity | 95.4% | 100.0% | 80.8% | 100.0% |

### fact種別別retention

| 条件 | percentage | numeric | categorical |
|---|---|---|---|
| C_high | 100.0% | 91.7% | 97.4% |
| D_identity | 100.0% | 92.1% | 97.4% |

**いずれの軸でもidentityがhighを下回る項目はなく、むしろ僅かに上回っている。**

## paired比較 (同一probe・同一seed、180ペア)

| 指標 | 値 |
|---|---|
| win (identity>high) | 7 |
| tie | 169 |
| loss (identity<high) | 4 |
| 平均delta(identity-high) | +0.2pt |
| broad平均delta | +0.3pt |
| narrow平均delta | 0.0pt |

## 9. 目視確認

**4件のloss全件を目視確認した:**

| probe/seed | high recall | identity recall | 内容 |
|---|---|---|---|
| V1-A/seed43 | 100% | 85.7% | 数値・%は完全保持、「RT確定」のみ欠落 |
| V1-A/seed46 | 100% | 85.7% | 同上 |
| V1-B/seed45 | 100% | 85.7%(実質満点) | 「550G〜1350Gまで」という自然な範囲表現による評価器の偽陰性(850Gが個別トークンとして現れないだけ) |
| V5-D/seed45 | 100% | 57.1%(実質満点) | 「出目Pの示唆はL」という自然な語順変化による評価器の偽陰性(「示唆L」という並び順を要求していたため) |

**真の壊滅的省略(P01で見られたような、%を全て捨ててゲーム数だけ残す挙動)は1件も再現しなかった。** 7件のwinケースも同一V1ファミリー("RT確定"の有無)を中心に分布しており、双方向のノイズ（時にhighが欠落、時にidentityが欠落）であり、identityへの一方的な劣化ではないことを確認した。

## 10. Identity維持確認（再解析のみ、新規学習・新規生成なし）

Phase4Uの既存結果ファイルを再解析し、以下が再現可能な集計であることを確認した。

| 指標 | 値 |
|---|---|
| naming probe genuine wrong-name率 | 4.1% |
| naming probe 正しい「リル」率 | 19.8% |
| E02 genuine wrong-name率 | 0.0% |
| E02 正しい「リル」率 | 25.0% |
| E36 genuine wrong-name率 | 5.0% |
| E36 placeholder率 | 0.0% |

## 15. 最終報告で答えるべき事項への回答

1. **P01回帰はheld-outでも再現したか** — いいえ。held-out broadquestionではhighとidentityはほぼ同等(96.9% vs 96.7%)。
2. **broad質問全体でidentityはhighより悪いか** — いいえ。broad質問のmean recallはidentity 95.4% > high 95.1%。
3. **差は何ポイントか** — +0.2pt(identity優位)、broadのみで+0.3pt。
4. **complete_answer_rateはどう変わったか** — 85.6%→87.2%へ改善。
5. **narrow質問では差があるか** — ない。両条件とも100.0%。
6. **percentage/mapping等の特定構造依存か** — いいえ。percentage(100%=100%)、numeric(91.7%→92.1%)、categorical(97.4%=97.4%)いずれもidentityが下回らない。
7. **「について教えて」という表現依存か** — 明確な依存は見られない。broad_topic/overview/explain/tell_me_allの4カテゴリ全てでidentityはhighと同等以上。
8. **P01は局所問題か一般化した問題か** — **局所問題**。同一トピック(天井)のheld-out variantでも系統的な悪化は見られず、seed依存のノイズと考えられる。
9. **identity改善を維持する価値はあるか** — **ある**。誤名乗り改善は極めて大きく、RAG性能への一般的な悪影響は確認されなかった。
10. **次に再学習する科学的根拠があるか** — 現時点では**ない**。P01単体の局所的な結果だけを理由に教師データを追加/調整する必要性は、本フェーズの結果によって支持されなかった。
11. **ある場合、何だけを変更すべきか** — 該当なし(根拠なしのため)。
12. **ratio-high-identityを最終候補として扱えるか** — **扱える**。ただしP01自体は元の値のまま気になる場合は、より広いheld-outでの追加確認や、実運用でのモニタリングを推奨する。

## 13. 禁止事項の遵守

Phase4V内で学習・candidate作成・identity教師変更・complex教師追加・ratio変更・LoRA変更・system prompt変更・merge・GGUF化・Git commit/pushは一切行っていない。

## 14. 完了確認

- pytest: **126 passed**
- protected assets: v4/ratio-high/ratio-high-identity adapter SHA-256、system.jinja2 MD5 全て不変
- git diff: 既存追跡ファイルへの差分なし
- git status: 新規作成ファイルのみ(`??`)、Git操作は実施していない

## 作成ファイル一覧

- `training/riru/eval/phase4v_probes.py`
- `training/riru/eval/phase4v_comprehensive_eval.py` / `phase4v_comprehensive_results.json`
- `training/riru/eval/phase4v_analyze.py`
- `training/riru/reports/phase4v_broad_question_analysis.json`
- `training/riru/reports/phase4v_paired_analysis.json`
- `training/riru/reports/_phase4v_losses_utf8.txt`
- `training/riru/reports/phase4v_summary.md`（本ファイル）

## 停止

評価・分析・目視確認・pytest・保護対象資産確認・レポート作成が完了しました。merge/GGUF化・正式採用・追加学習・Phase 4W等への自動移行・Git commit/pushは一切行っていません。次のご判断をお待ちします。
