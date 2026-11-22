import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faqbot.knowledge import KnowledgeBase, evaluate  # noqa: E402
from faqbot.storage import Storage, normalize_phone  # noqa: E402

KB_PATH = ROOT / "data" / "knowledge_base.json"
CASES = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_file(KB_PATH, backend="tfidf")


# ----- knowledge base -----

def test_exact_question_is_found(kb):
    for entry in json.loads(KB_PATH.read_text()):
        assert kb.answer(entry["question"]).entry_id == entry["id"]


def test_typo_tolerance(kb):
    assert kb.answer("how long does delivry take").entry_id == "delivery_time"
    assert kb.answer("payement methods").entry_id == "payment"


def test_low_confidence_returns_none(kb):
    assert kb.answer("hello") is None
    assert kb.answer("   ") is None


def test_search_is_ranked(kb):
    scores = [m.score for m in kb.search("payment by card", top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_empty_kb_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([], backend="tfidf")


def test_tfidf_baseline_does_not_regress(kb):
    """Guards the measured baseline (69% on the eval set); fails if a KB edit makes things worse."""
    report = evaluate(kb, CASES)
    assert report["accuracy"] >= 0.65, report["misses"]


# ----- storage -----

@pytest.mark.parametrize("raw, expected", [
    ("+38 (067) 123-45-67", "+380671234567"),
    ("0671234567", "0671234567"),
    ("12345", None),
    ("call me", None),
])
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_storage(tmp_path):
    st = Storage(str(tmp_path / "t.db"))
    assert st.add_order(1, "Ann", "+380671234567", "1 kg dark, espresso") == 1
    assert st.orders()[0]["name"] == "Ann"
    for q in ["Do you sell tea?", "do you sell tea?", "gift cards?"]:
        st.log_unanswered(1, q, 0.1)
    top = st.top_unanswered()
    assert top[0] == {"question": "do you sell tea?", "times": 2}


# ----- bot handlers (real aiogram FSM, fake Telegram messages) -----

@pytest.fixture
def bot_module(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_BACKEND", "tfidf")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "bot.db"))
    monkeypatch.setenv("ADMIN_CHAT_ID", "999")
    sys.modules.pop("bot", None)
    import bot
    return bot


def fake_message(text: str, user_id: int = 42):
    return SimpleNamespace(
        text=text,
        from_user=SimpleNamespace(id=user_id, full_name="Test User", username="tester"),
        chat=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
    )


@pytest.fixture
def fsm():
    from aiogram.fsm.context import FSMContext
    from aiogram.fsm.storage.base import StorageKey
    from aiogram.fsm.storage.memory import MemoryStorage

    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42))


async def test_order_flow(bot_module, fsm):
    tg = SimpleNamespace(send_message=AsyncMock())
    await bot_module.order_start(fake_message(bot_module.BTN_ORDER), fsm)
    await bot_module.order_name(fake_message("Ann"), fsm)

    bad = fake_message("not a phone")
    await bot_module.order_phone(bad, fsm)
    assert "doesn't look like a phone" in bad.answer.call_args.args[0]
    assert await fsm.get_state() == bot_module.OrderForm.phone.state

    await bot_module.order_phone(fake_message("+38 067 123 45 67"), fsm)
    done = fake_message("1 kg dark roast, espresso grind")
    await bot_module.order_item(done, fsm, tg)

    assert await fsm.get_state() is None
    assert "Order #1" in done.answer.call_args.args[0]
    order = bot_module.storage.orders()[0]
    assert (order["name"], order["phone"]) == ("Ann", "+380671234567")
    admin_chat, admin_text = tg.send_message.call_args.args
    assert admin_chat == 999 and "Order #1" in admin_text


async def test_faq_answer_and_handover(bot_module):
    tg = SimpleNamespace(send_message=AsyncMock())
    msg = fake_message("is shipping free?")
    await bot_module.faq(msg, tg)
    assert "free for orders over $40" in msg.answer.call_args.args[0]
    tg.send_message.assert_not_called()

    unknown = fake_message("hello")
    await bot_module.faq(unknown, tg)
    assert "passed your question to a manager" in unknown.answer.call_args.args[0]
    assert bot_module.storage.top_unanswered()[0]["question"] == "hello"
    tg.send_message.assert_called_once()
