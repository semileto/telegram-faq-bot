"""Telegram support bot for a fictional coffee shop "Bean & Brew".

- answers FAQ from a knowledge base (offline retrieval, optional LLM phrasing)
- hands over to a manager when unsure and logs the question
- collects order requests (name -> phone -> item) and notifies the admin chat
- /stats for the admin: orders count and most frequent unanswered questions

Env vars: BOT_TOKEN (required), ADMIN_CHAT_ID, KB_BACKEND (auto|embeddings|tfidf), OPENAI_API_KEY (all optional).
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup, ReplyKeyboardRemove

from faqbot.knowledge import KnowledgeBase, rephrase_with_llm
from faqbot.storage import Storage, normalize_phone

ROOT = Path(__file__).parent


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from .env into os.environ (real env vars win). Keeps secrets out of the code."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:  # empty placeholders (ADMIN_CHAT_ID=) mean "not set"
                os.environ.setdefault(key.strip(), value)


if __name__ == "__main__":  # only when started as the bot, so tests never pick up a local .env
    load_env_file(ROOT / ".env")

BTN_FAQ, BTN_ORDER, BTN_MANAGER = "❓ Ask a question", "🛒 Place an order", "👤 Talk to a manager"
MENU = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
    [KeyboardButton(text=BTN_FAQ), KeyboardButton(text=BTN_ORDER)],
    [KeyboardButton(text=BTN_MANAGER)],
])

kb = KnowledgeBase.from_file(ROOT / "data" / "knowledge_base.json", backend=os.getenv("KB_BACKEND", "auto"))
storage = Storage(os.getenv("DB_PATH", str(ROOT / "bot.db")))
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0")) or None
router = Router()


class OrderForm(StatesGroup):
    name = State()
    phone = State()
    item = State()


async def notify_admin(bot: Bot, text: str) -> None:
    if ADMIN_CHAT_ID:
        await bot.send_message(ADMIN_CHAT_ID, text)


@router.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Hi! I'm the Bean & Brew assistant ☕\nAsk me anything about delivery, payment, "
                         "our coffee, or place an order.", reply_markup=MENU)


@router.message(Command("cancel"))
async def cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Cancelled.", reply_markup=MENU)


@router.message(Command("id"))
async def chat_id(message: Message) -> None:
    """Shows the chat id, so the owner can put it into ADMIN_CHAT_ID."""
    await message.answer(f"Chat id: {message.chat.id}")


@router.message(F.text == BTN_FAQ)
async def faq_hint(message: Message) -> None:
    await message.answer("Just type your question, e.g. «how long is delivery?»")


@router.message(F.text == BTN_MANAGER)
async def manager(message: Message, bot: Bot) -> None:
    user = message.from_user
    await notify_admin(bot, f"👤 {user.full_name} (@{user.username}, id {user.id}) asks for a manager.")
    await message.answer("A manager will message you shortly (Mon-Sat, 9:00-20:00).")


# ----- order flow -----

@router.message(F.text == BTN_ORDER)
async def order_start(message: Message, state: FSMContext) -> None:
    await state.set_state(OrderForm.name)
    await message.answer("What's your name? (/cancel to stop)", reply_markup=ReplyKeyboardRemove())


@router.message(OrderForm.name, F.text)
async def order_name(message: Message, state: FSMContext) -> None:
    await state.update_data(name=message.text.strip()[:100])
    await state.set_state(OrderForm.phone)
    await message.answer("Your phone number?")


@router.message(OrderForm.phone, F.text)
async def order_phone(message: Message, state: FSMContext) -> None:
    phone = normalize_phone(message.text)
    if not phone:
        await message.answer("That doesn't look like a phone number. Example: +380671234567")
        return
    await state.update_data(phone=phone)
    await state.set_state(OrderForm.item)
    await message.answer("What would you like? (coffee, roast, grind, amount)")


@router.message(OrderForm.item, F.text)
async def order_item(message: Message, state: FSMContext, bot: Bot) -> None:
    data = await state.update_data(item=message.text.strip()[:500])
    await state.clear()
    order_id = storage.add_order(message.from_user.id, data["name"], data["phone"], data["item"])
    await notify_admin(bot, f"🛒 Order #{order_id}\n{data['name']}, {data['phone']}\n{data['item']}")
    await message.answer(f"Thank you! Order #{order_id} received, we'll call you to confirm.", reply_markup=MENU)


# ----- admin -----

@router.message(Command("stats"))
async def stats(message: Message) -> None:
    if message.chat.id != ADMIN_CHAT_ID:
        logging.info("/stats ignored: chat %s is not ADMIN_CHAT_ID %s", message.chat.id, ADMIN_CHAT_ID)
        return
    lines = [f"Orders: {len(storage.orders())}", "Top unanswered questions:"]
    lines += [f"• {r['question']} ({r['times']}×)" for r in storage.top_unanswered()] or ["• none"]
    await message.answer("\n".join(lines))


# ----- free text -> FAQ -----

@router.message(F.text)
async def faq(message: Message, bot: Bot) -> None:
    match = kb.answer(message.text)
    if match is None:
        best = kb.search(message.text, top_k=1)
        storage.log_unanswered(message.from_user.id, message.text, best[0].score if best else 0.0)
        await notify_admin(bot, f"❔ Unanswered from {message.from_user.full_name}: {message.text}")
        await message.answer("I'm not sure about that one. I've passed your question to a manager, "
                             "they'll reply soon.", reply_markup=MENU)
        return
    await message.answer(await rephrase_with_llm(message.text, match))


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    bot = Bot(os.environ["BOT_TOKEN"])
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
