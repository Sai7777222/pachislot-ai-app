"""RAG (Retrieval-Augmented Generation) 層。

DESIGN.md の RAG 層 (embedder / retriever / context_builder / pipeline) に対応。
Phase 3 時点ではシンプルなベクトル検索のみ (BM25・リランカー・エージェント検索は未実装)。

このパッケージのどのモジュールも、チャット回答処理の中で外部ネットワークへ
アクセスしない (Embedding はローカルモデル、Vector DB はローカル永続化のみ)。
"""
