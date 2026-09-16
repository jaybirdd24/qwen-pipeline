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


def test_japanese_quotes_and_unspaced_sentences() -> None:
    text = "「こんにちは！」次の文です。最後です？"
    assert chunk_text(text, "ja", maximum=10) == ["「こんにちは！」", "次の文です。", "最後です？"]
    assert chunk_text(text, "ja", maximum=100) == [text]
    long_text = "あいうえお" * 10
    assert "".join(chunk_text(long_text, "ja", maximum=12)) == long_text
    assert all(len(chunk) <= 12 for chunk in chunk_text(long_text, "ja", maximum=12))


def test_korean_preserves_word_spaces_and_sentence_boundaries() -> None:
    text = "작은 새가 노래해요. 아이가 웃어요! 함께 집에 가요."
    chunks = chunk_text(text, "ko", maximum=15)
    assert chunks == ["작은 새가 노래해요.", "아이가 웃어요!", "함께 집에 가요."]
    assert " ".join(chunks) == text
    assert chunk_text("작은 새가 나무 위에서 노래해요", "ko", maximum=9) == [
        "작은 새가 나무",
        "위에서 노래해요",
    ]


def test_portuguese_preserves_accents_and_word_spaces() -> None:
    text = "A lua brilha. Que coração feliz! Vamos para casa."
    chunks = chunk_text(text, "pt", maximum=20)
    assert chunks == ["A lua brilha.", "Que coração feliz!", "Vamos para casa."]
    assert " ".join(chunks) == text
