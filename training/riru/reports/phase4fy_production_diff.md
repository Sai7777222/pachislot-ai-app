# Phase4FY Section6/29: 本番コード変更差分(`phase4fy_production_diff.md`)

**重要: 以下の変更は `git add`/`commit` されていません。**Section27の指示通り、CASE判定と
人間によるレビュー確認が完了するまでcommitを保留しています。

## 変更ファイル一覧

| ファイル | 種別 | 変更内容 |
|---|---|---|
| `src/pachislot_ai/rag/entity_attribution.py` | 新規 | Phase4FXオフラインプロトタイプの production port (query entity抽出・title_match_score・title補完検索・entity/evidence binding・no-evidence合成チャンク・`select_grounded_chunks()`) |
| `src/pachislot_ai/rag/vector_store.py` | 変更 | `VectorStore.get_all()` 追加(title補完検索用の既存Chromaコレクション一括取得。新しい検索器・新しいembeddingではない) |
| `src/pachislot_ai/rag/retriever.py` | 変更 | `Retriever.get_all_chunks()` 追加(`get_all()`のラップ + 既存`search()`と同じ正規化テキストによる重複除去) |
| `src/pachislot_ai/rag/pipeline.py` | 変更 | `RagPipeline.build_context()` 内、chunk再取得ブロック直後・`build_rag_context()`呼び出し直前に entity-aware selection の1呼び出しを追加 |
| `tests/unit/test_entity_attribution.py` | 新規 | Section13 unit tests (16件) |

## 変更していないもの(frozen、確認済み)

- `config/prompts/system.jinja2` — 未変更(read onlyで内容確認のみ)
- `config/prompts/rag_context.jinja2` — 未変更。既存の空context fallback文言
  (「該当する構造化データ・解説文章は登録されていません」)をentity-attribution層の
  0件選別ケースがそのまま利用する設計。
- `src/pachislot_ai/services/chat_service.py` — 未変更
- `src/pachislot_ai/rag/structured_lookup.py` — 未変更(構造化DB検索ロジックはentity-attribution
  と独立)
- `src/pachislot_ai/rag/context_builder.py` — 未変更
- `src/pachislot_ai/rag/embedder.py` — 未変更(embeddingモデル・base retrieverロジックは無changed)
- `src/pachislot_ai/rag/vector_store.py` の `query()`/`upsert_chunks()`/`delete_by_machine()` —
  既存メソッドは無変更、`get_all()`のみ追加
- training/inference/adapter関連ファイル一切(Phase4ZG hashは末尾の`phase4fy_end_integrity.json`
  で確認)
- dispatch/Policy C3相当のロジック — 補足: 現在の `src/pachislot_ai` 本番コードには
  dispatch/Policy C3に相当するモジュールは実装されていない(この概念は
  `training/riru/guard/phase4zr_conservative_dispatch.py` 等の診断ハーネス内にのみ存在する)。
  したがって本フェーズはこれらのファイルに一切触れておらず、「semantics unchanged」は
  自明に満たされる。詳細は `phase4fy_routing.json` を参照。

## 挿入ポイント(`RagPipeline.build_context()`)

base retrieval(`self._retriever.search`) → 機種確定後のchunk再取得(`refined`) →
**[NEW] title supplemental retrieval + entity attribution + query/entity binding
(`select_grounded_chunks`)** → `build_rag_context()`(production prompt rendering) →
generation(呼び出し元 `ChatService`)。

dispatch/Policy C3(存在する場合の呼び出し元)より前でも、generation後でもなく、
Section6で指定された通りの位置(retrieval〜prompt renderingの間)に挿入されている。

## 完全なdiff

```diff
$(git diff の内容は下記ファイルを参照)
```

完全なunified diffは `training/riru/reports/_phase4fy_diff_raw.txt` に保存済み
(git diff の生出力そのまま)。要点は上表と挿入ポイント図の通り。
