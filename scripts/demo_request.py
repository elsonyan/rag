import argparse

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a demo request to the local RAG API.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--mode", choices=("chat", "rag"), default="rag", help="Request mode")
    parser.add_argument("--message", default="电话是多少？我没有你们的货币怎么办？", help="User input")
    args = parser.parse_args()

    endpoint = "/chat/" if args.mode == "chat" else "/chat/rag"
    payload = {"messages": [args.message]} if args.mode == "chat" else {"query": args.message}

    response = requests.post(
        f"{args.base_url}{endpoint}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
