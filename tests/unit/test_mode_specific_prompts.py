"""Phase4FC4 Section21: mode-specific system prompt選択の unit tests。"""

from __future__ import annotations

from pachislot_ai.core.config import Settings
from pachislot_ai.llm.base import ChatMessage
from pachislot_ai.services.chat_service import ChatService


class _DummyLLM:
    model_name = "dummy"

    async def chat(self, *a, **k):  # pragma: no cover
        pass

    async def chat_stream(self, *a, **k):  # pragma: no cover
        pass

    async def health_check(self):  # pragma: no cover
        return True


def _make_service() -> ChatService:
    settings = Settings()
    return ChatService(_DummyLLM(), settings.resolved_system_prompt_path, None)


# 1. SMALL_TALK selects conversational prompt
def test_small_talk_selects_small_talk_prompt():
    svc = _make_service()
    prompt = svc._select_system_prompt([ChatMessage(role="user", content="おはよう！")])
    assert "雑談" in prompt
    assert prompt != svc._system_prompt


# 2. IDENTITY selects identity prompt
def test_identity_selects_identity_prompt():
    svc = _make_service()
    prompt = svc._select_system_prompt([ChatMessage(role="user", content="君の名前は？")])
    assert "名前" in prompt and "自己紹介" in prompt
    assert prompt != svc._system_prompt


# 3. OOD selects boundary prompt
def test_ood_selects_boundary_prompt():
    svc = _make_service()
    prompt = svc._select_system_prompt([ChatMessage(role="user", content="今日の天気を教えて")])
    assert "専門" in prompt
    assert prompt != svc._system_prompt


# 4. FACTUAL retains existing prompt
def test_factual_retains_original_system_prompt():
    svc = _make_service()
    prompt = svc._select_system_prompt([ChatMessage(role="user", content="天井は何ゲームですか")])
    assert prompt == svc._system_prompt


# 5. UNKNOWN retains RAG path (uses original system prompt, not a mode-specific one)
def test_unknown_retains_original_system_prompt():
    svc = _make_service()
    prompt = svc._select_system_prompt(
        [ChatMessage(role="user", content="GGとSGGの違いを初心者向けに説明して")]
    )
    assert prompt == svc._system_prompt


# 6. mode prompt not persisted into history (each call is independent, computed fresh)
def test_mode_prompt_not_persisted_across_calls():
    svc = _make_service()
    p1 = svc._select_system_prompt([ChatMessage(role="user", content="おはよう！")])
    p2 = svc._select_system_prompt([ChatMessage(role="user", content="天井は何ゲームですか")])
    p3 = svc._select_system_prompt([ChatMessage(role="user", content="君の名前は？")])
    assert p1 != p2
    assert p2 != p3
    assert p1 != p3
    # 呼び出し順序に関わらず、同じqueryなら常に同じ結果(ステートレス)
    assert svc._select_system_prompt([ChatMessage(role="user", content="おはよう！")]) == p1


# 7. small-talk prompt explicitly instructs against DB fallback wording (the phrase
#    itself legitimately appears as a quoted negative example within a "don't say
#    this" instruction — checking bare absence would be a false-negative test, since
#    the prompt's whole point is to name and prohibit exactly that phrasing).
def test_small_talk_prompt_prohibits_db_fallback_wording():
    svc = _make_service()
    prompt = svc._mode_system_prompts["SMALL_TALK"]
    assert "登録データにありません" in prompt  # 禁止例として明示的に言及されている
    assert "はせず" in prompt or "しないで" in prompt  # 否定・禁止の指示であることを確認
    assert "自然に答えて" in prompt  # 代わりに求める振る舞いも明示されている


# 8. identity path answers using product identity context (name is リル, explicitly)
def test_identity_prompt_establishes_canonical_name():
    svc = _make_service()
    prompt = svc._mode_system_prompts["IDENTITY_PERSONA"]
    assert "リル" in prompt


# 9. factual path prompt unchanged (byte-identical to loaded system.jinja2)
def test_factual_system_prompt_unchanged():
    from jinja2 import Template
    settings = Settings()
    svc = _make_service()
    expected = Template(settings.resolved_system_prompt_path.read_text(encoding="utf-8")).render()
    assert svc._system_prompt == expected


# 10. no mode prompt stacking: _build_messages produces exactly one system message
#     for non-RAG modes (no rag_context to append)
def test_no_mode_prompt_stacking_for_small_talk():
    svc = _make_service()
    messages = svc._build_messages([ChatMessage(role="user", content="おはよう！")], None)
    system_messages = [m for m in messages if m.role == "system"]
    assert len(system_messages) == 1


# 11. small-talk: no RAG (rag_context=None passed through _build_messages produces
#     a single system message, the small-talk one)
def test_small_talk_build_messages_has_only_small_talk_system_message():
    svc = _make_service()
    messages = svc._build_messages([ChatMessage(role="user", content="趣味とかあるの？")], None)
    assert messages[0].role == "system"
    assert "雑談" in messages[0].content
    assert len(messages) == 2  # system + user


# 12. factual RAG still enabled: _build_messages appends rag_context on top of the
#     unchanged factual system prompt when one is provided
def test_factual_build_messages_appends_rag_context_to_original_prompt():
    from pachislot_ai.rag.context_builder import RagContext
    svc = _make_service()
    fake_ctx = RagContext(
        prompt_text="【検索結果】ダミー", structured_source_ids=[], structured_sources=[],
        chunk_sources=[], is_empty=False,
    )
    messages = svc._build_messages(
        [ChatMessage(role="user", content="天井は何ゲームですか")], fake_ctx
    )
    assert messages[0].content == svc._system_prompt
    assert messages[1].role == "system"
    assert messages[1].content == "【検索結果】ダミー"
    assert len(messages) == 3  # system(factual) + system(rag context) + user
