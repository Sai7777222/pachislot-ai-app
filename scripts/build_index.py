"""RAGストア (rag_store.db) のドキュメントをチャンク化 -> Embedding -> Vector DB 投入する CLI。

「外部サイトからデータを取得・更新する処理」に相当し、チャット回答処理からは
呼び出されない。Embedding もオフラインのローカルモデルのみを使用する。

例:
    python scripts/build_index.py --machine-id smart_million_god_kamigami_no_kiseki
    python scripts/build_index.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlalchemy import select  # noqa: E402

from pachislot_ai.core.config import get_settings  # noqa: E402
from pachislot_ai.data.db import create_rag_engine, open_session  # noqa: E402
from pachislot_ai.data.models.rag import RagDocument  # noqa: E402
from pachislot_ai.rag.chunking import chunk_rag_document  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGストアをチャンク化してVector DBへ投入する")
    parser.add_argument("--machine-id", default=None, help="対象機種ID (未指定時は --all が必要)")
    parser.add_argument("--all", action="store_true", help="rag_store.db 内の全機種を対象にする")
    args = parser.parse_args()

    if not args.machine_id and not args.all:
        print("ERROR: --machine-id または --all のいずれかを指定してください")
        return 1

    settings = get_settings()
    rag_engine = create_rag_engine(settings.rag_db_path)

    with open_session(rag_engine) as session:
        stmt = select(RagDocument)
        if args.machine_id:
            stmt = stmt.where(RagDocument.machine_id == args.machine_id)
        docs = list(session.scalars(stmt))

    if not docs:
        print("対象のRAGドキュメントが見つかりませんでした。")
        return 1

    machine_ids = sorted({d.machine_id for d in docs})
    print(f"Target machines: {machine_ids}")
    print(f"RAG documents: {len(docs)}")

    print(
        f"Loading embedding model: {settings.embedding_model_path} "
        f"(device={settings.embedding_device})"
    )
    start = time.perf_counter()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    print(
        f"Embedding model loaded in {time.perf_counter() - start:.2f}s "
        f"(dim={embedder.dimension})"
    )

    store = VectorStore(settings.vector_db_path, settings.vector_db_collection)

    for machine_id in machine_ids:
        store.delete_by_machine(machine_id)

    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_rag_document(doc, max_chars=settings.rag_chunk_max_chars))

    print(f"Chunks to embed: {len(all_chunks)}")

    start = time.perf_counter()
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embeddings = embedder.embed_documents([c["text"] for c in batch])
        store.upsert_chunks(batch, embeddings)
        print(f"  embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")
    print(f"Embedding + index build took {time.perf_counter() - start:.2f}s")

    print(f"Vector DB path: {settings.vector_db_path}")
    print(f"Collection: {settings.vector_db_collection}")
    print(f"Total vectors in collection: {store.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
