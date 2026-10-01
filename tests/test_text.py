import pytest

from stories import text

# --- Tokenizing ---------------------------------------------------------------------------


def test_tokenize_lowercases_stems_and_drops_stopwords() -> None:
    assert text.tokenize("Tell me about a time you MENTORED someone") == ["mentor", "someon"]


def test_tokenize_splits_on_punctuation_hyphens_and_underscores() -> None:
    assert text.tokenize("cross-team, data_driven!") == ["cross", "team", "data", "driven"]


def test_tokenize_keeps_non_ascii_words() -> None:
    assert text.tokenize("café") == ["café"]


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])
def test_tokenize_drops_contractions_and_possessive_s(apostrophe: str) -> None:
    sentence = "What's a weakness you've had? I didn't know Rust's syntax, and we'll see.".replace("'", apostrophe)

    # Contractions are function words, and a possessive is the word it's attached to.
    assert text.tokenize(sentence) == ["weak", "know", "rust", "syntax", "see"]


def test_tokenize_keeps_apostrophes_inside_names() -> None:
    assert text.tokenize("O'Brien") == ["obrien"]


# --- Word overlap ---------------------------------------------------------------------------


def test_word_overlap_is_the_share_of_the_shorter_texts_words_in_the_other() -> None:
    # "deadline" and "mentor" are shared, out of the shorter text's three words.
    overlap = text.word_overlap("missed deadline mentor", "a mentor who set a deadline and a budget")

    assert overlap == pytest.approx(2 / 3)


def test_word_overlap_ignores_stopwords_and_word_forms() -> None:
    assert text.word_overlap("I mentored them", "mentoring") == 1.0


def test_word_overlap_with_no_meaningful_words_is_zero() -> None:
    assert text.word_overlap("the and of", "mentor") == 0.0
