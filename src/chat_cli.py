"""Interactive CLI: chat with Dostoevsky, full pipeline wired in (Stages 3-6).

Each message runs: query rewrite -> hybrid retrieval -> LLM-judge rerank ->
diversity cap -> stance-aware dossier -> persona answer, with conversation memory.

Usage:
    python src/chat_cli.py            # persona voice
    python src/chat_cli.py --debug    # also print rewritten query + retrieved chunks
Commands:  /quit  /reset  /debug
"""
import sys

import conversation as C

BANNER = (
    "Chatting with Dostoevsky — grounded in *White Nights*.\n"
    "Commands: /quit  /reset (clear memory)  /debug (toggle retrieval trace)\n"
)


def main():
    history: list[dict] = []
    debug = "--debug" in sys.argv
    print(BANNER)
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg == "/reset":
            history.clear()
            print("(memory cleared)\n")
            continue
        if msg == "/debug":
            debug = not debug
            print(f"(debug {'on' if debug else 'off'})\n")
            continue

        out = C.chat(history, msg)
        if debug:
            short = [c.split("white_nights_")[-1] for c in out["retrieved"]]
            print(f"  [rewritten: {out['rewritten']}]")
            print(f"  [retrieved: {short}]")
        print(f"\nDostoevsky> {out['reply']}\n")


if __name__ == "__main__":
    main()
