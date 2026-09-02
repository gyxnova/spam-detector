"""
Optional LLM layer: given a message + classifier verdict, asks Groq
for a short natural-language explanation of why it looks like spam/ham.

Requires GROQ_API_KEY in environment (.env).
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            raise RuntimeError('GROQ_API_KEY not set in environment (.env)')
        _client = Groq(api_key=api_key)
    return _client


def explain(text: str, label: str, confidence: float = None) -> str:
    """Returns a 1-2 sentence explanation of the classifier verdict."""
    client = _get_client()
    conf_str = f' (confidence: {confidence:.0%})' if confidence is not None else ''

    prompt = (
        f'A spam classifier labeled this message as \'{label}\'{conf_str}.\n\n'
        f'Message: "{text}"\n\n'
        'In 1-2 short sentences, explain the specific cues (wording, urgency, '
        'links, requests, etc.) that likely drove this verdict. Be concise and concrete.'
    )

    response = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=400,
        temperature=0.3,
    )

    content = response.choices[0].message.content
    print(f"[explain debug] finish_reason={response.choices[0].finish_reason}, content={content!r}")

    return (content or "").strip()