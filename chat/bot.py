import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
import os
from func import get_answer_from_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TOKEN")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer("Привет, напишите ваш вопрос, и я попробую найти ответ")


@dp.message()
async def handle_message(message: Message):
    user_question = message.text
    logger.info(f"Получен вопрос от пользователя: {user_question}")

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    llm_response = get_answer_from_llm(user_question)

    if llm_response is None:
        await message.answer("Извините, я не могу ответить на этот вопрос.")
    else:
        logger.info({llm_response})
        await message.answer(llm_response)