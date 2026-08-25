"""共通 pytest フィクスチャ。

`llm` マーカーが付いたテストは実モデル (LLM_MODEL_PATH) のロードを伴うため、
GPU/モデルが利用できる開発機でのみ実行する想定。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from pachislot_ai.main import create_app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """アプリ全体 (lifespan で実際に LLM をロード) を一度だけ起動して使い回す。"""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
