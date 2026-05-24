"""
AI generation service using the Anthropic API.
Generates: notes, flashcards, quizzes, podcast scripts, mindmaps.
"""
import json
import anthropic
from ..core.config import get_settings

settings = get_settings()


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _truncate(text: str, max_chars: int = 40_000) -> str:
    """Avoid blowing the context window on massive docs."""
    return text[:max_chars] + ("\n\n[... document truncated ...]" if len(text) > max_chars else "")


# ── NOTES ────────────────────────────────────────────────────────────────────

def generate_notes(text: str, title: str) -> dict:
    prompt = f"""You are an expert study assistant. Given the document below, produce clear, well-structured study notes.

Return ONLY valid JSON in this exact shape (no markdown, no preamble):
{{
  "summary": "2-3 sentence overview",
  "key_concepts": [
    {{"term": "...", "definition": "..."}}
  ],
  "sections": [
    {{"heading": "...", "bullets": ["..."]}}
  ],
  "takeaways": ["..."]
}}

Document title: {title}
---
{_truncate(text)}"""

    msg = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()
    return json.loads(raw)


# ── FLASHCARDS ────────────────────────────────────────────────────────────────

def generate_flashcards(text: str, title: str, count: int = 15) -> dict:
    prompt = f"""You are an expert tutor. Create {count} high-quality spaced-repetition flashcards from the document below.

Return ONLY valid JSON (no markdown, no preamble):
{{
  "flashcards": [
    {{"front": "question or term", "back": "answer or definition", "difficulty": "easy|medium|hard"}}
  ]
}}

Document title: {title}
---
{_truncate(text)}"""

    msg = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(msg.content[0].text.strip())


# ── QUIZ ─────────────────────────────────────────────────────────────────────

def generate_quiz(text: str, title: str, count: int = 10) -> dict:
    prompt = f"""You are an exam-writing expert. Generate {count} multiple-choice questions from the document below.

Return ONLY valid JSON (no markdown, no preamble):
{{
  "quiz": [
    {{
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct": "A",
      "explanation": "Why this is correct"
    }}
  ]
}}

Document title: {title}
---
{_truncate(text)}"""

    msg = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(msg.content[0].text.strip())


# ── PODCAST SCRIPT ────────────────────────────────────────────────────────────

def generate_podcast_script(text: str, title: str) -> dict:
    prompt = f"""You are a podcast producer. Turn the document below into an engaging, conversational study podcast script between two hosts: Alex (lead explainer) and Sam (curious student).

Return ONLY valid JSON (no markdown, no preamble):
{{
  "title": "Episode title",
  "duration_estimate": "~X minutes",
  "script": [
    {{"speaker": "Alex", "line": "..."}},
    {{"speaker": "Sam", "line": "..."}}
  ]
}}

Document title: {title}
---
{_truncate(text, 20000)}"""

    msg = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(msg.content[0].text.strip())


# ── MIND MAP ──────────────────────────────────────────────────────────────────

def generate_mindmap(text: str, title: str) -> dict:
    prompt = f"""You are a knowledge-mapping expert. Build a hierarchical mind map from the document below.

Return ONLY valid JSON (no markdown, no preamble):
{{
  "root": "{title}",
  "branches": [
    {{
      "topic": "Main topic",
      "subtopics": [
        {{"label": "...", "detail": "optional one-liner"}}
      ]
    }}
  ]
}}

Document title: {title}
---
{_truncate(text)}"""

    msg = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(msg.content[0].text.strip())


# ── CHAT ─────────────────────────────────────────────────────────────────────

def chat_with_document(text: str, title: str, messages: list[dict]) -> str:
    system = f"""You are a helpful study assistant. Answer questions about the document below clearly and concisely.
Only use information from the document. If the answer isn't in the document, say so.

Document title: {title}
---
{_truncate(text)}"""

    response = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=messages
    )
    return response.content[0].text
