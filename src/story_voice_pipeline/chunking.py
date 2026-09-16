"""Deterministic language-aware text chunking."""

from __future__ import annotations

import re

from .config import CHUNKING_VERSION
from .errors import PipelineError

_SENTENCE_PATTERNS = {
    "en": re.compile(r".*?(?:[.!?]+(?:[\"'”’)]*)|$)(?:\s+|$)", re.DOTALL),
    "zh": re.compile(r".*?(?:[。！？]+(?:[\"'”’）)]*)|$)", re.DOTALL),
    "es": re.compile(r".*?(?:[.!?]+(?:[\"'”’)]*)|$)(?:\s+|$)", re.DOTALL),
    "fr": re.compile(r".*?(?:[.!?]+(?:[\"'”’)]*)|$)(?:\s+|$)", re.DOTALL),
    "de": re.compile(r".*?(?:[.!?]+(?:[\"'”’)]*)|$)(?:\s+|$)", re.DOTALL),
}

# Preserve existing languages' boundaries (and cache keys). Japanese punctuation
# may directly follow a closing quote; Korean and Portuguese retain word spaces.
_SENTENCE_PATTERNS["ja"] = re.compile(r".*?(?:[。！？!?]+[」』”’）)]*|$)", re.DOTALL)
_SENTENCE_PATTERNS["ko"] = _SENTENCE_PATTERNS["en"]
_SENTENCE_PATTERNS["pt"] = _SENTENCE_PATTERNS["en"]

_SPACE_SEPARATED_LANGUAGES = {"en", "es", "fr", "de", "ko", "pt"}


def _sentences(paragraph: str, language: str) -> list[str]:
    matches = [match.group(0).strip() for match in _SENTENCE_PATTERNS[language].finditer(paragraph)]
    return [match for match in matches if match]


def _hard_split(text: str, maximum: int, language: str) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > maximum:
        boundary = maximum
        if language in _SPACE_SEPARATED_LANGUAGES:
            whitespace = remaining.rfind(" ", 1, maximum + 1)
            if whitespace > 0:
                boundary = whitespace
        pieces.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_text(text: str, language: str, maximum: int = 400) -> list[str]:
    """Split text while preferring paragraphs, then complete sentences."""
    if language not in _SENTENCE_PATTERNS:
        raise PipelineError(f"Unsupported chunking language: {language}")
    if maximum < 1:
        raise PipelineError("Maximum chunk size must be positive")
    text = text.strip()
    if not text:
        raise PipelineError("Cannot chunk empty text")

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not paragraph:
            continue
        paragraph_sentences = _sentences(paragraph, language)
        units.extend(paragraph_sentences or [paragraph])
        units.append("")  # paragraph boundary marker

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not unit:
            if current:
                chunks.append(current)
                current = ""
            continue
        candidates = _hard_split(unit, maximum, language) if len(unit) > maximum else [unit]
        for candidate in candidates:
            separator = " " if current and language in _SPACE_SEPARATED_LANGUAGES else ""
            if current and len(current) + len(separator) + len(candidate) > maximum:
                chunks.append(current)
                current = candidate
            else:
                current = f"{current}{separator}{candidate}" if current else candidate
    if current:
        chunks.append(current)
    return chunks


__all__ = ["CHUNKING_VERSION", "chunk_text"]
