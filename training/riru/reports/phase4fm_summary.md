# Phase4FM 最終報告書
## Product Moderation Layer Integration & Validation

1. **CASE**: FM-A(Moderation Integration Successful)
2. **FC4 commit hash**: `7cf6f8e17b2461f107d7eb6041505f231e6e1155`
3. **FC4 push結果**: 成功(`8c36025..7cf6f8e checkpoint/identity-closure-phase4zn-baseline -> checkpoint/identity-closure-phase4zn-baseline`)
4. **モデレーションアーキテクチャ**: 決定的ルールベース(LLM/外部API不使用)。入力チェック(dispatch/RAG/生成より前)と出力チェック(生成後・表示より前)の2段構成。`src/pachislot_ai/moderation/`(normalize.py/policy.py/matcher.py/engine.py)。`ChatService.check_input()`/`check_output()`として統合。
5. **ポリシーconfig配置**: `config/moderation.yaml`(コードから分離、YAML形式)
6. **正規化戦略**: NFKC正規化 + ASCII小文字化 + 空白正規化(全match_form共通の基礎) + opt-inの区切り記号除去(normalized_sequenceのみ)。詳細は`phase4fm_normalization.json`。
7. **match modes**: `exact`(全体一致) / `token_boundary`(語境界安全な部分一致、漢字・カタカナ・英数字の連続文字種チェック) / `normalized_sequence`(難読化対応の部分一致、opt-in)
8. **input policy種別**: `HARD_BLOCK` / `ALLOW`(2値。Section4のA-D分類はinput/output両ポリシーの組み合わせから導出)
9. **output policy種別**: `HARD_BLOCK` / `ALLOW`(同上)
10. **input blocked test件数**: unit 8件(exact/NFKC/whitespace/punctuation/token_boundary embedded/boundary safety等) + integration 4パターン(パラメタライズ)
11. **input bypass to RAG**: 0
12. **input bypass to LLM**: 0
13. **output blocked test件数**: unit 6件 + integration 4パターン(パラメタライズ) + streaming 1件
14. **output visible leak**: 0
15. **streaming戦略**: 生成完了までバッファリングし、モデレーション判定後に単一deltaとして送出(Section12の明示的推奨方針、複雑な逐次検閲は非実装)
16. **streaming visible leak**: 0(SSE生行全体を検索して確認)
17. **fallback echo件数**: 0(safe_responseにルールID・カテゴリ・一致語のいずれも一切含まれない)
18. **benign near-match FP**: 0
19. **pachislot FP**: 0(GG/SGG/天国ロング/ガイアベル等の代表語彙で確認、加えて本番regression50件全件でFP=0)
20. **small-talk regression**: 0(hedge 0/20、FC4baseline 0/65と一致)
21. **identity regression**: 0(「君の名前は？」→「リルだよ！」維持、既知の受容済みwrong-name caseもFC4とbyte-identical)
22. **OOD regression**: 0(境界維持、既知の4/15 UNKNOWN経路hedgeはFC4から不変・受容済み)
23. **Q6**: FC4とbyte-identical(drift=0)
24. **AT-F**: FC4とbyte-identical(drift=0)
25. **RT-A/RT-B**: FC4とbyte-identical(drift=0)
26. **SGG/GG準備中**: FC4とbyte-identical(drift=0)
27. **GG中**: FC4とbyte-identical(drift=0)
28. **天国ロング**: FC4とbyte-identical(drift=0)
29. **AD-04**: FC4とbyte-identical(drift=0)
30. **P02**: FC4とbyte-identical(drift=0)
31. **P04**: FC4とbyte-identical(drift=0、LOW status継続許容)
32. **LC-08**: FC4とbyte-identical(drift=0)
33. **Q11**: FC4とbyte-identical(drift=0)
34. **Q15**: FC4とbyte-identical(drift=0)
35. **Q17**: FC4とbyte-identical(drift=0)
36. **critical factual regression**: 0(known_failure12全12件・RAG8全8件、いずれも100% byte-identical to FC4)
37. **dispatch drift**: 0/260(vs FC4)
38. **dangerous misroute**: 0/260
39. **multi-turn contamination**: なし(5シナリオ確認、生成された禁止表現がhistoryへ混入する経路は構造的に存在しない。詳細は`phase4fm_multiturn.json`)
40. **追加テスト数**: 40件(unit 26件 + integration 14件)、既存テスト1件を意図的仕様変更に合わせて更新
41. **pytest開始/終了**: 306 passed → 346 passed(0 failed、306+40=346で一致)
42. **モデレーションオーバーヘッド**: check_input 0.00737ms/call、check_output 0.00821ms/call、合計0.016ms/call(目標<1msを大幅にクリア)
43. **streaming latency影響**: 追加のtime-to-first-visible-response 平均約5.4秒(最大43.7秒、生成完了まで何も表示されなくなるため。詳細は`phase4fm_performance.json`)
44. **Phase4ZG無変更**: SHA256ハッシュ`278fe7ae...9dcc9dc`が生成前後で完全一致、訓練は一切行っていない
45. **factual prompt無変更**: `system.jinja2`のgit diff無し
46. **conversation prompts無変更**: `small_talk.jinja2`/`identity_persona.jinja2`/`ood_boundary.jinja2`のgit diff無し
47. **dispatch無変更**: `src/pachislot_ai/dispatch/`のgit diff無し、GT260 drift=0で実証
48. **FY/FZ無変更**: `entity_attribution.py`/`structured_lookup.py`/`evidence_arbitration.py`のgit diff無し
49. **DB無変更**: 構造化DB/RAG DBともに変更なし
50. **embedding無変更**: embeddingモデル・vector DBともに変更なし
51. **trainingなし**: 本フェーズはLoRA/SFT/QLoRAを一切実行していない
52. **external moderation APIなし**: 外部API呼び出しは一切行っていない(決定的ルールマッチングのみ)
53. **quantizationなし**: GGUF/Q8/Q5化は未実施(継続してSection29の前提条件待ち)
54. **Slack**: 送信予定(下記テンプレート参照、Webhook URLは非公開)
55. **moderation ACCEPT/REJECT**: ACCEPT
56. **Final Product Candidate可能 YES/NO**: YES(次フェーズとして推奨)
57. **recommended next phase**: Final Product Candidate Comprehensive Re-test
58. **auto-startなし**: 次フェーズは自動開始しない。ここで停止する。

---

## 補足: 生成予算

Stage A相当の事前計算(retrieval/dispatchのみ、生成なし) + 実生成58件(単発50件+multi-turn安全ターン8件) = 実生成合計58件。上限(mandatory<=100, preferred<=60)を両方満たす。

## 補足: 誠実な留保事項

- Phase4FM-MT-04のmulti-turnシナリオでは、Phase4ZGが禁止マーカーを自然にechoしなかったため、そのライブ生成ではoutput-block自体は発火しなかった。output-blockパスの直接証明はFakeLLMProviderによる制御された出力を用いたunit/integration testで行っている(Section18の明示的指示通り)。
- streamingの逐次表示体験は失われた(生成完了まで何も表示されない)。これはSection12の明示的な安全性優先の指示に基づく意図的なトレードオフである。

---

## Slack通知テンプレート(PASS)

```
リル Phase4FM Moderation Integration完了
CASE: FM-A
input block bypass: 0
output prohibited leak: 0
stream leak: 0
pachislot false positive: 0
factual regression: 0
conversation regression: 0
次: Final Product Candidate Re-test
詳細: phase4fm_summary.md
```

Stop.
