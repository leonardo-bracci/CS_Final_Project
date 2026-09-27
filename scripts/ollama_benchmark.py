"""
ollama_benchmark.py
-------------------
Compares llama3.2 with qwen2.5:3b on the same sample emotion summary,
measuring time to first token and total generation time.

Usage (from the project root):
    ollama pull llama3.2
    ollama pull qwen2.5:3b
    python scripts/ollama_benchmark.py
"""

import time

MODELS_TO_COMPARE = ["llama3.2", "qwen2.5:3b"]

SYSTEM_PROMPT = """You are a supportive, privacy-preserving wellbeing assistant.
You are NOT a therapist and must NEVER diagnose a medical or psychiatric
condition. Your job is to reflect back the emotional patterns you observe
(facial expression, spoken tone, and the words used) in a warm, non-clinical
way, and suggest general wellbeing practices. If the observed patterns
suggest significant or persistent distress, gently and clearly recommend the
user speak to a mental health professional. Keep responses concise."""

SAMPLE_PROMPT = (
    "Here is what was observed during the session:\n\n"
    "Facial emotions per scene:\n"
    "- threatening scene: fear (5 readings)\n"
    "- joyful scene: happy (8 readings)\n"
    "- sad scene: sad (6 readings)\n"
    "- relaxing scene: neutral (10 readings)\n\n"
    "What the user said: \"I've been feeling pretty stressed about work lately, "
    "but today was okay.\"\n"
    "Emotional tone of their words: fear (confidence 0.6)\n\n"
    "Based on this, write a short, warm psychological reflection: "
    "(1) a summary of the emotional patterns you notice, "
    "(2) one or two focus areas the user might benefit from working on, "
    "(3) two or three concrete, general wellbeing recommendations."
)


def benchmark_model(model_name):
    """Run the sample prompt through one model, timing first-token and total
    generation time using Ollama's streaming API."""
    import ollama

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_PROMPT},
    ]

    print(f"\n=== {model_name} ===")
    start = time.perf_counter()
    first_token_time = None
    full_reply = ""

    try:
        stream = ollama.chat(model=model_name, messages=messages, stream=True)
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            full_reply += chunk["message"]["content"]
    except Exception as e:
        print(f"[{model_name}] failed - is it pulled? Run: ollama pull {model_name}")
        print(f"Error: {e}")
        return None

    total_time = time.perf_counter() - start

    print(f"Time to first token: {first_token_time:.2f}s")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Reply length: {len(full_reply)} characters")
    print(f"\n--- Reply ---\n{full_reply}\n")

    return {
        "model": model_name,
        "first_token_s": round(first_token_time, 2) if first_token_time else None,
        "total_time_s": round(total_time, 2),
        "reply": full_reply,
    }


if __name__ == "__main__":
    results = []
    for model in MODELS_TO_COMPARE:
        result = benchmark_model(model)
        if result:
            results.append(result)

    print("\n\n=== Summary ===")
    print(f"{'Model':<20} | {'First token (s)':>16} | {'Total time (s)':>15}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} | {r['first_token_s']:>16} | {r['total_time_s']:>15}")
