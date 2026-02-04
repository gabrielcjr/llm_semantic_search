import os
from dotenv import load_dotenv
import anthropic
from search import build_prompt

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat(question: str) -> str:
    """Send question to Claude with RAG context."""
    prompt = build_prompt(question)

    message = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL"),
        max_tokens=1024,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text


def main():
    print("RAG Chat with Claude (type 'exit' to quit)")
    print("-" * 50)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("sair", "exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        response = chat(question)
        print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()