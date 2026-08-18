from __future__ import annotations

from story_voice_pipeline.chunking import chunk_text


def test_english_prefers_sentences_and_keeps_punctuation() -> None:
    text = "One short sentence. “A quoted sentence stays whole!” Last one?"
    chunks = chunk_text(text, "en", maximum=40)
    assert chunks == ["One short sentence.", "“A quoted sentence stays whole!”", "Last one?"]
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_mandarin_sentence_boundaries() -> None:
    text = "第一句话。第二句话！第三句话？"
    assert chunk_text(text, "zh", maximum=20) == ["第一句话。第二句话！第三句话？"]
    assert chunk_text(text, "zh", maximum=8) == ["第一句话。", "第二句话！", "第三句话？"]


def test_paragraphs_are_not_combined() -> None:
    assert chunk_text("First.\n\nSecond.", "en", maximum=100) == ["First.", "Second."]


def test_oversized_english_sentence_splits_at_whitespace() -> None:
    chunks = chunk_text("alpha beta gamma delta epsilon.", "en", maximum=20)
    assert chunks == ["alpha beta gamma", "delta epsilon."]
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_oversized_mandarin_sentence_hard_splits() -> None:
    chunks = chunk_text("这是一个没有任何中间标点而且特别长的中文句子。", "zh", maximum=20)
    assert len(chunks) == 2
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_spanish_uses_space_separated_sentence_chunking() -> None:
    text = "Una frase corta. ¿Otra frase completa? La última."
    assert chunk_text(text, "es", maximum=25) == [
        "Una frase corta.",
        "¿Otra frase completa?",
        "La última.",
    ]
