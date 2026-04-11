"""
Quick test: verify OPENAI_API_KEY loads and gpt-4o-mini responds.

Usage:
    python test_openai.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY not found.")
    print("Make sure your .env file exists and contains:")
    print("  OPENAI_API_KEY=sk-...")
    exit(1)

# Mask key for display: show first 8 and last 4 chars only
masked = api_key[:8] + "..." + api_key[-4:]
print(f"API key loaded: {masked}")

# Prompt gpt-4o-mini
print("Sending test prompt to gpt-4o-mini...")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.3,
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {
            "role": "user",
            "content": "A student asks: what is 12 × 8? Give a one-sentence Socratic hint, not the answer.",
        },
    ],
)

reply = response.choices[0].message.content
print(f"\nModel response:\n{reply}")
print(f"\nTokens used: {response.usage.total_tokens}")
print("\nAll good — API key works and gpt-4o-mini is responding.")
