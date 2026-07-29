import asyncio
import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import ingest
import rag_engine
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4000

WELCOME_MESSAGE = (
    "Hi! Send me a PDF document, then ask me questions about it and I'll answer "
    "using only what's in it.\n\n"
    "Your documents are private to this chat.\n\n"
    "Commands:\n"
    "/reset - forget the PDFs you've sent and start over"
)


async def _reply_long(update: Update, text: str):
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        await update.message.reply_text(text[i : i + TELEGRAM_MAX_LEN])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    chat_id = update.effective_chat.id

    await update.message.reply_text(f"Got {document.file_name}, reading it...")

    loop = asyncio.get_running_loop()
    try:
        telegram_file = await document.get_file()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / document.file_name
            await telegram_file.download_to_drive(str(file_path))
            chunk_count = await loop.run_in_executor(
                None, ingest.ingest_pdf, chat_id, file_path
            )
    except Exception:
        logger.exception("Failed to ingest document")
        await update.message.reply_text(
            "Sorry, I couldn't process that PDF. Please try again."
        )
        return

    await update.message.reply_text(
        f"Added {document.file_name} ({chunk_count} chunks). Ask away!"
    )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if not question:
        return

    chat_id = update.effective_chat.id
    await update.message.chat.send_action("typing")

    loop = asyncio.get_running_loop()
    try:
        answer, sources = await loop.run_in_executor(
            None, rag_engine.ask, chat_id, question
        )
    except rag_engine.VectorStoreNotReadyError as e:
        await update.message.reply_text(str(e))
        return
    except Exception:
        logger.exception("Failed to answer question")
        await update.message.reply_text("Something went wrong answering that. Please try again.")
        return

    lines = [answer, "", "Sources:"]
    seen = set()
    for source in sources:
        key = (source["source"], source["page"])
        if key in seen:
            continue
        seen.add(key)
        name = Path(source["source"]).name if source["source"] else "unknown"
        lines.append(f"- {name} (page {source['page']})")

    await _reply_long(update, "\n".join(lines))


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rag_engine.reset_user, chat_id)

    await update.message.reply_text("Done — I've forgotten your PDFs. Send a new one to start over.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception", exc_info=context.error)


def build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    application.add_error_handler(on_error)

    return application


def main():
    application = build_application()
    logger.info("Starting Telegram bot (polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
