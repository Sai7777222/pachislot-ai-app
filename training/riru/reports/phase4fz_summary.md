# Phase4FZ 最終レポート — Structured Facts Entity-Binding Closure

**CASE判定: FZ-A(Structured Facts Closure Successful)**

以下、Section24が要求する42項目に沿って回答する。

1. **CASE**: FZ-A (Structured Facts Closure Successful)
2. **FY commit hash**: `b61ee8c` (`feat: add entity-aware RAG context assembly`, 32ファイル)
3. **FY push status**: ローカルcommit完了・**push未完了**。許可システムによりpushコマンドがブロックされ、ユーザーへ`git push origin checkpoint/identity-closure-phase4zn-baseline`の実行を依頼中。
4. **structured root cause**: `find_relevant_structured_facts()`(SQLベース、以前は無変更)が、metric_facts の `dimensions_json` 内 `group` キー等の値(天国/継続/終了等、136種の異なるmetric_key間で使い回される汎用stateラベル)を、生のquery文字列に対し境界チェック無しの単純substring test(`v in query`)で照合していたため、(a) 「天国」が「天国ロング」の一部として、(b) 「終了」が「終了後」の一部として偶然一致し、無関係なテーブルのデータが混入していた。zoneのzone_key/name/aliasesも同様の素朴なsubstring testを使っており、「GG」が「SGG」の部分文字列として誤って一致する既知パターンも未対策だった。
5. **structured schema usable YES/NO**: YES — metric_key(136種、chunkのtitleに相当する概念識別子)とzone_key/name/aliases(8zone)は概念を一意に識別できる既存フィールドとして利用可能。新しいフィールド・スキーマ変更は不要だった。
6. **binding architecture**: SF-A(Exact/Boundary Binding)を採用。Phase4FX/FYの`extract_query_entities()`・`title_match_score()`をそのまま再利用してmetric_keyとの照合を行い、加えて新設した`_value_matches_query_with_boundary()`(漢字/カタカナ/ハイフンを継続文字とみなす対称的境界チェック)でdimension値・zone名の照合を行う二段構成とした。SF-B(parent-aware)は該当schema無しのため不採用。
7. **GT hash**: `40010d2b71fa9e6d82a1ab4b870d54067e1f1c4a5c464a1358cc223814570ffa`(40件: phantom15/real20/close_concept5、DB実データのみに基づく客観判定)
8. **天国ロング result**: 修正前は実在の「天国」モードデータを流用し「規定回数1〜3回、確率0.332/0.332/0.336」という具体的な架空説明を生成していたが、修正後は`登録データにありません。`と完全にdecline。構造化facts=0件を確認。
9. **AT-F result**: 修正前は無関係な「ガイアステージ終了抽選」テーブルの`0.129`が混入していたが、修正後は`登録データにその情報は見つからないんだ。`と完全にdecline(chunk側・structured側とも0件)。
10. **RT-A/RT-B result**: 修正前後とも0件・完全decline(元々このケースでは構造化facts側の漏れは無かったが、回帰していないことを確認)。
11. **GG継続関連 result**: 修正前は無関係な「ガイアステージ終了抽選」テーブルのgroup=継続データがGG関連chunkと混在していたが、修正後は「継続契機ごとのSGGゲーム数振り分け」等、真にGG/SGG関連のmetric_keyのみが選ばれ、無関係テーブルの混入は解消された。
12. **phantom structured misbinding**: 0/15(GT frozen、DB実データで客観判定)
13. **unsupported numeric**: 0/59(全生成probe中、grounded数値監査でflagged=1件のみ、それはP04の自己計算差分でLOW/スコープ外)
14. **real structured recall**: 20/20(GT real全件でEXPECTED_FACTS通り非空)
15. **completeness regression**: 0件。むしろAD-04(「ヤメ時はいつがいい？」)でmetric_key「ヤメ時」が新たに正しく回収され、以前の完全declineからgrounded回答へ改善する副次的benefitを確認。
16. **combined-source misattribution**: 0/59(chunk safe/structured safe 32件、chunk empty/structured safe 4件、chunk safe/structured empty 3件、both empty 20件、複数実在entity11件の全カテゴリで確認)
17. **Q6 result**: PASS。全数値(50G/7枚/G/350枚/10〜100G/75%以上)は構造化データに原文表記のまま存在、grounded。
18. **P02 result**: PASS。実DBに「ボーナス確率」データ無し、正しくdecline(変化なし)。
19. **P04 result**: PASS(条件付き)。97.2%/114.6%はgrounded、「17.4%」は自己計算(LOW、Section15の指示通り今回は未修正)。
20. **LC-08 result**: PASS。修正確認: 以前の`0.129`漏れが解消され完全decline。
21. **Q11 result**: PASS。天井側は全数値grounded、ヤメ時側は正直にdecline(以前の曖昧な推測的言及から明確なdeclineに改善)。
22. **Q15 result**: PASS。Phase4FYで修正済みの挙動(機種概要chunk使用+正直な情報不足開示)を維持。
23. **Q17 result**: PASS。変化なし、grounded。
24. **AD-04 result**: PASS(completeness改善)。以前は完全declineだったが、修正後はmetric_key「ヤメ時」を正しく回収し「GG終了後…G-ZONE終了後、32G消化」というgrounded回答に改善。
25. **GG中 result**: PASS。既存のtitle補完検索(Phase4FX/FY、無変更)経由で継続して正しく回収されることを確認(本フェーズはこの経路に触れていない)。
26. **tests added**: 15件(`tests/unit/test_structured_facts_binding.py`)
27. **pytest start/end**: start=252 passed → end=267 passed(0 failed、0 regressions)
28. **added latency**: 新規ロジック単体で約1.6ms(DB fetch除く)、find_relevant_structured_facts()全体では約5.6ms(DB fetch込み、DB fetch自体は既存オーバーヘッド)。<1ms推奨目標をわずかに超過するが、mandatoryではなく実際の生成latency(数秒〜数十秒)と比較すると無視できる水準。
29. **generation count**: 59件(既知失敗8件+RAG50必須8件+FZ固有必須3件+GT40件)。予算上限180、目標120未満に対し余裕を持って収まった。
30. **Phase4ZG unchanged**: 確認済み(`278fe7ae...`、Phase4ZH以降一貫して同一hash)
31. **prompt unchanged**: 確認済み(`system.jinja2`・`rag_context.jinja2`とも無変更)
32. **chunk entity logic unchanged**: 確認済み(`entity_attribution.py`・`pipeline.py`・`retriever.py`・`vector_store.py`のgit statusに本フェーズでの差分なし)
33. **DB unchanged**: 確認済み(ingest/upsert等は一切実行していない)
34. **embedding unchanged**: 確認済み(Embedder・呼び出し方法とも無変更)
35. **trainingなし**: 確認済み(訓練は一切実施していない)
36. **quantizationなし**: 確認済み(GGUF/Q8/Q5関連の作業は一切実施していない)
37. **LOW arithmetic risk status**: 今回も意図的に未修正のまま記録のみ(P04/PT-08、入力値grounded・計算結果は正確、Final Candidateで再確認予定)
38. **Slack**: この応答の直後にPASSテンプレートで送信する
39. **structured facts fix ACCEPT/REJECT**: **ACCEPT** — Section9-14の全必須gateがPASS、フェーズ内で発見した2件の実装バグも修正・検証済み
40. **Final Candidate可能 YES/NO**: **YES** — RAG factual safety closure candidateとして推奨。ただし本番コードのcommitは人間確認後に実施すること(Section21の指示通り自動commitしていない)。
41. **recommended next phase**: Final Candidate Comprehensive Re-test(Phase4FY・FZの両変更をまとめて対象とする)。その前に、(a) Phase4FYのcommit `b61ee8c` のpush完了、(b) Phase4FZの`structured_lookup.py`変更のレビュー・commit承認、の2点をユーザーに確認いただくことを推奨する。
42. **auto-startなし**: 本レポートの提出をもって停止する。次フェーズは自動開始しない。

---

## 未commitの変更(人間確認待ち)

- `src/pachislot_ai/rag/structured_lookup.py`(修正 — Section6-8のentity-binding修正)
- `tests/unit/test_structured_facts_binding.py`(新規 — Section16、15テスト)
- `training/riru/guard/phase4fz_root_cause_audit.py`・`phase4fz_build_gt.py`・`phase4fz_gt_verification.py`・`phase4fz_precompute_contexts.py`(診断・検証スクリプト)
- `training/riru/guard/run_phase4fz_generation.py`(GPU生成スクリプト)
- `training/riru/reports/phase4fz_*.json`・`.txt`(本フェーズの全成果物)

ここで停止する。
