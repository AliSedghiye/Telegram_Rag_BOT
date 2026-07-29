from rag_engine import ask, VectorStoreNotReadyError

# Local terminal testing uses its own collection, separate from any Telegram chat.
# Feed it with: python src/ingest.py <path-to-pdf>
CHAT_ID = "cli"


def main():
    while True:
        question = input("\nAsk a question, or type 'exit': ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        try:
            answer, sources = ask(CHAT_ID, question)
        except VectorStoreNotReadyError as e:
            print(f"\n{e}")
            continue

        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        for source in sources:
            print("-", source["source"], "page:", source["page"])


if __name__ == "__main__":
    main()
