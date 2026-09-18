"""
CLI script to quickly test the RAG pipeline and agent without running the web server.
Run:  python test_agent.py
"""
import sys
from agent import get_agent

TEST_QUESTIONS = [
    "How can I reduce plastic use at home?",
    "What eco-friendly travel options are available in cities?",
    "What government grants are available for solar panels in the UK?",
    "How do I start composting at home?",
    "What is my carbon footprint and how can I reduce it?",
]

def main():
    print("=" * 70)
    print("  🌿 Eco Lifestyle Agent — CLI Test")
    print("=" * 70)

    agent = get_agent()

    questions = TEST_QUESTIONS
    if len(sys.argv) > 1:
        # Allow passing a custom question as a CLI argument
        questions = [" ".join(sys.argv[1:])]

    for question in questions:
        print(f"\n{'─' * 70}")
        print(f"❓ {question}")
        print("─" * 70)
        result = agent.chat(question)
        print(result["answer"])
        print(f"\n📚 Sources: {', '.join(result['sources'])}")
        print(f"📄 Context chunks used: {result['context_chunks']}")

    print(f"\n{'=' * 70}")
    print("Test complete.")


if __name__ == "__main__":
    main()
