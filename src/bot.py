import asyncio
import html
import logging
from datetime import datetime
from pathlib import Path

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import ingest
import ingest_manager
import rag_engine
import storage_cleanup
from config import (
    MAX_CHUNKS_PER_USER,
    MAX_PDF_SIZE_MB,
    TELEGRAM_ADMIN_IDS,
    TELEGRAM_BOT_TOKEN,
    UPLOAD_TTL_HOURS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# httpx/telegram log every request at INFO level, including the full URL —
# which contains the bot token (".../bot<TOKEN>/getMe"). Silencing these
# keeps the token out of container logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Per-user guard against a single user flooding the shared thread-pool
# executor with many concurrent question requests.
_in_flight_questions: set[int] = set()

TELEGRAM_MAX_LEN = 4000
CLEANUP_INTERVAL_SECONDS = 60 * 60

WELCOME_MESSAGE = (
    "👋 <b>Welcome!</b>\n\n"
    "Send me a PDF and I'll add it to your own knowledge base. Once it's in, just "
    "ask me anything about it and I'll answer using only what's in your documents "
    "— never anyone else's.\n\n"
    "🗑 Your file itself is deleted from the server right after it's processed — "
    "only the extracted text is kept.\n"
    f"📦 To keep storage manageable, you can store up to {MAX_CHUNKS_PER_USER} chunks "
    "of content — if you go over, I'll automatically drop your oldest document and "
    "let you know.\n\n"
    "Use /documents, /upload, /status, or /help any time."
)

UPLOAD_HELP_TEXT = (
    "📤 <b>How to upload</b>\n\n"
    "Just send a PDF file directly in this chat — as a document, not a photo. "
    "I'll ask you to confirm before adding it to your knowledge base. Once "
    "confirmed, ask me anything about it!\n\n"
    f"⏳ If you upload but don't confirm, I'll auto-delete the file after "
    f"{UPLOAD_TTL_HOURS:g}h to save space."
)


async def _reply_long(update: Update, text: str, parse_mode: str | None = None):
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        await update.message.reply_text(text[i : i + TELEGRAM_MAX_LEN], parse_mode=parse_mode)


def _is_authorized(update: Update) -> bool:
    if not TELEGRAM_ADMIN_IDS:
        return True
    user = update.effective_user
    return bool(user and user.id in TELEGRAM_ADMIN_IDS)


def _format_status_text(status: dict) -> str:
    if status["running"]:
        return "⏳ Ingestion is currently running."
    if status["last_run_status"] is None:
        return "🆕 No ingestion has run yet. Send me a PDF to build your knowledge base."
    if status["last_run_status"] == "success":
        text = f"✅ Last ingestion succeeded ({status['last_run_chunks']} new chunks)."
        evicted = status.get("last_run_evicted") or []
        if evicted:
            names = ", ".join(html.escape(n) for n in evicted)
            text += f"\n⚠️ Evicted oldest document(s) to stay within your storage cap: {names}"
        return text
    return f"❌ Last ingestion failed: {html.escape(str(status['last_run_error']))}"


def _format_documents(user_id: int) -> tuple[str, InlineKeyboardMarkup | None]:
    docs = rag_engine.list_documents(user_id)
    if not docs:
        return "📭 You haven't added any documents yet. Send me a PDF to get started.", None

    lines = ["📚 <b>Your documents</b>", ""]
    keyboard = []
    for d in docs:
        added = datetime.fromtimestamp(d["ingested_at"]).strftime("%Y-%m-%d %H:%M")
        name = html.escape(d["doc_name"])
        lines.append(f"📄 <b>{name}</b> — {d['chunks']} chunks, added {added}")
        short_name = d["doc_name"][:24]
        keyboard.append(
            [InlineKeyboardButton(f'🗑 Delete "{short_name}"', callback_data=f"delete_{d['doc_id']}")]
        )
    return "\n".join(lines), InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE, parse_mode="HTML")


async def documents_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, keyboard = _format_documents(update.effective_user.id)
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = ingest_manager.get_status(update.effective_user.id)
    await update.message.reply_text(_format_status_text(status), parse_mode="HTML")


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if not question:
        return

    user_id = update.effective_user.id

    if user_id in _in_flight_questions:
        await update.message.reply_text(
            "⏳ Still working on your last question — please wait for that to finish."
        )
        return

    _in_flight_questions.add(user_id)
    try:
        await update.message.chat.send_action("typing")

        loop = asyncio.get_running_loop()
        try:
            answer, sources = await loop.run_in_executor(None, rag_engine.ask, question, user_id)
        except rag_engine.VectorStoreNotReadyError as e:
            await update.message.reply_text(f"📭 {e}")
            return
        except rag_engine.SystemBusyError as e:
            await update.message.reply_text(f"⏳ {e}")
            return
        except Exception:
            logger.exception("Failed to answer question")
            await update.message.reply_text("⚠️ Something went wrong answering that. Please try again.")
            return

        lines = [f"🤖 <b>Answer</b>\n{html.escape(answer)}", "", "📎 <b>Sources</b>"]
        seen = set()
        for source in sources:
            key = (source["source"], source["page"])
            if key in seen:
                continue
            seen.add(key)
            name = source.get("doc_name") or (Path(source["source"]).name if source["source"] else "unknown")
            lines.append(f"• {html.escape(name)} (page {source['page']})")

        await _reply_long(update, "\n".join(lines), parse_mode="HTML")
    finally:
        _in_flight_questions.discard(user_id)


async def _run_ingest_job(message, user_id, job):
    """job is a zero-arg callable, e.g. a run(user_id) or run_file(path, user_id) closure."""

    async def runner():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, job)

        status = ingest_manager.get_status(user_id)
        if status["last_run_status"] == "success":
            reply = f"✅ Done ({status['last_run_chunks']} new chunks). You can ask questions now."
            evicted = status.get("last_run_evicted") or []
            if evicted:
                names = ", ".join(html.escape(n) for n in evicted)
                reply += (
                    f"\n\n⚠️ To stay within your storage limit, I removed your oldest "
                    f"document(s) to make room: {names}. You can no longer ask about "
                    "them — resend if you need them again."
                )
            await message.reply_text(reply, parse_mode="HTML")
        else:
            await message.reply_text(
                f"❌ Ingestion failed: {html.escape(str(status['last_run_error']))}", parse_mode="HTML"
            )

    asyncio.create_task(runner())


async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("🚫 You are not authorized to run this command.")
        return

    user_id = update.effective_user.id
    if not any(ingest.user_dir(user_id).glob("*.pdf")):
        await update.message.reply_text(
            f"📭 No PDFs waiting to be added yet.\n\n{UPLOAD_HELP_TEXT}", parse_mode="HTML"
        )
        return

    if not ingest_manager.try_start(user_id):
        await update.message.reply_text(
            "⏳ Another ingestion is running right now (system-wide), please try again shortly."
        )
        return

    await update.message.reply_text("⏳ Starting ingestion, this may take a while...")
    await _run_ingest_job(update.message, user_id, lambda: ingest_manager.run(user_id))


def _safe_filename(raw_name: str | None) -> str:
    """Reduce an attacker-controlled filename to a single safe path component.

    Telegram's file_name is arbitrary attacker-supplied text — without this,
    a name like "../../../../etc/cron.d/evil" would let pathlib's `/` operator
    split on the embedded "/" characters and write outside the user's upload
    directory entirely (verified: it resolves clean out of Data/raw/<uid>/).
    """
    name = Path(raw_name or "document.pdf").name  # drops any directory components
    name = name.strip().lstrip(".") or "document.pdf"
    return name[-150:]  # keep filesystem-safe length


async def handle_pdf_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("🚫 You are not authorized to upload documents.")
        return

    document = update.message.document
    user_id = update.effective_user.id

    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    if document.file_size and document.file_size > max_bytes:
        await update.message.reply_text(
            f"📏 That file is too large ({document.file_size / 1024 / 1024:.1f} MB) — "
            f"the limit is {MAX_PDF_SIZE_MB:g} MB."
        )
        return

    user_pdf_dir = ingest.user_dir(user_id)
    user_pdf_dir.mkdir(parents=True, exist_ok=True)
    # Unique name so two uploads with the same filename can't clobber each other
    # while they're waiting for confirmation.
    safe_name = _safe_filename(document.file_name)
    unique_name = f"{document.file_unique_id}_{safe_name}"
    file_path = user_pdf_dir / unique_name

    # Defense in depth: confirm the resolved path really is inside the user's dir.
    if user_pdf_dir.resolve() not in file_path.resolve().parents:
        await update.message.reply_text("⚠️ That filename isn't allowed — please rename the file and resend.")
        return

    telegram_file = await document.get_file()
    await telegram_file.download_to_drive(str(file_path))
    context.user_data["pending_pdf"] = file_path

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Yes, add it", callback_data="ingest_confirm"),
                InlineKeyboardButton("🚫 Not now", callback_data="ingest_cancel"),
            ]
        ]
    )
    await update.message.reply_text(
        f'📄 Got "{html.escape(document.file_name)}". Add it to your knowledge base so you can ask '
        "questions about it? Only you can query documents you've added. The file is "
        "deleted from the server right after — only its extracted text is kept.",
        parse_mode="HTML",
        reply_markup=keyboard,
    )


async def handle_other_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📎 I can only read PDF files right now — please send a PDF.")


async def handle_ingest_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    if query.data == "ingest_cancel":
        await query.edit_message_text(
            "👍 Okay, I kept the file for now but won't add it yet. Send /upload whenever "
            f"you're ready — unconfirmed files are auto-deleted after {UPLOAD_TTL_HOURS:g}h "
            "to save space."
        )
        return

    pdf_path = context.user_data.get("pending_pdf")
    if not pdf_path or not pdf_path.exists():
        await query.edit_message_text("⚠️ I don't have that file anymore — please resend it.")
        return

    if not _is_authorized(update):
        await query.edit_message_text("🚫 You are not authorized to build the knowledge base.")
        return

    if not ingest_manager.try_start(user_id):
        await query.edit_message_text(
            "⏳ Another ingestion is running right now (system-wide), hang tight and try again shortly."
        )
        return

    await query.edit_message_text("⏳ Adding it to your knowledge base, this may take a while...")
    context.user_data.pop("pending_pdf", None)
    await _run_ingest_job(query.message, user_id, lambda: ingest_manager.run_file(pdf_path, user_id))


async def handle_delete_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    doc_id = query.data[len("delete_") :]

    doc_name = rag_engine.delete_document(user_id, doc_id)
    if doc_name:
        await query.edit_message_text(
            f"🗑 Deleted <b>{html.escape(doc_name)}</b>. You can no longer ask about it.",
            parse_mode="HTML",
        )
    else:
        await query.edit_message_text("⚠️ That document is already gone.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception", exc_info=context.error)


async def _cleanup_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        storage_cleanup.cleanup()
    except Exception:
        logger.exception("Storage cleanup failed")


async def _post_init(application: Application):
    await application.bot.set_my_commands(
        [
            BotCommand("documents", "List your uploaded documents"),
            BotCommand("upload", "Add pending PDFs / how to upload"),
            BotCommand("status", "Check last ingestion status"),
            BotCommand("help", "Show help"),
        ]
    )


def build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(_post_init).build()
    application.job_queue.run_repeating(_cleanup_job, interval=CLEANUP_INTERVAL_SECONDS, first=0)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("documents", documents_command))
    application.add_handler(CommandHandler("upload", upload_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf_document))
    application.add_handler(
        MessageHandler(filters.Document.ALL & ~filters.Document.PDF, handle_other_document)
    )
    application.add_handler(CallbackQueryHandler(handle_ingest_callback, pattern="^ingest_"))
    application.add_handler(CallbackQueryHandler(handle_delete_callback, pattern="^delete_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    application.add_error_handler(on_error)

    return application


def main():
    application = build_application()
    logger.info("Starting Telegram bot (polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
