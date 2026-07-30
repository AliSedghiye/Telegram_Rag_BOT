from rag_engine import ask, VectorStoreNotReadyError


def main():
    user_id = int(input("Telegram user id to query as: ").strip())

    while True:
        question = input("\nAsk a question, or type 'exit': ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        try:
            answer, sources = ask(question, user_id)
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
