import asyncio
import logging
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import ingest_manager
import rag_engine
from config import DATA_DIR, TELEGRAM_ADMIN_IDS, TELEGRAM_BOT_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4000

WELCOME_MESSAGE = (
    "Hi! Ask me anything about the mobility program documents and I'll answer "
    "using only what's in them.\n\n"
    "Commands:\n"
    "/ingest - rebuild the knowledge base from the source PDFs\n"
    "/status - check the last ingestion result"
)


async def _reply_long(update: Update, text: str):
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        await update.message.reply_text(text[i : i + TELEGRAM_MAX_LEN])


def _is_authorized(update: Update) -> bool:
    if not TELEGRAM_ADMIN_IDS:
        return True
    user = update.effective_user
    return bool(user and user.id in TELEGRAM_ADMIN_IDS)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE)


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if not question:
        return

    await update.message.chat.send_action("typing")

    loop = asyncio.get_running_loop()
    try:
        answer, sources = await loop.run_in_executor(None, rag_engine.ask, question)
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


async def ingest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("You are not authorized to run this command.")
        return

    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.pdf")):
        await update.message.reply_text(f"No PDF files found in {DATA_DIR}")
        return

    if not ingest_manager.try_start():
        await update.message.reply_text("Ingestion is already running.")
        return

    await update.message.reply_text("Starting ingestion, this may take a while...")

    async def run():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ingest_manager.run)

        status = ingest_manager.get_status()
        if status["last_run_status"] == "success":
            await update.message.reply_text(
                f"Ingestion complete ({status['last_run_chunks']} chunks)."
            )
        else:
            await update.message.reply_text(f"Ingestion failed: {status['last_run_error']}")

    asyncio.create_task(run())


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = ingest_manager.get_status()

    if status["running"]:
        await update.message.reply_text("Ingestion is currently running.")
    elif status["last_run_status"] is None:
        await update.message.reply_text("No ingestion has run yet. Send /ingest to build the knowledge base.")
    elif status["last_run_status"] == "success":
        await update.message.reply_text(
            f"Last ingestion succeeded ({status['last_run_chunks']} chunks)."
        )
    else:
        await update.message.reply_text(f"Last ingestion failed: {status['last_run_error']}")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception", exc_info=context.error)


def build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("ingest", ingest_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    application.add_error_handler(on_error)

    return application


def main():
    application = build_application()
    logger.info("Starting Telegram bot (polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
