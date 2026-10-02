import pytest

from sentiment.preprocessing import TweetPreprocessor, clean_tweet


def raw(text: str) -> str:
    return clean_tweet(text, remove_stopwords=False, stem=False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Check this http://t.co/abc out", "check this out"),
        ("@user hello there", "hello there"),
        ("I love #python", "i love python"),
        ("&quot;quoted&quot; &amp; done", "quoted done"),
        ("sooooo happy!!!", "soo happy"),
        ("good food", "good food"),
        ("2 days 4 me", "days me"),
        ("I don't like it", "i do not like it"),
        ("can't won't", "can not will not"),
    ],
)
def test_cleaning_steps(text, expected):
    assert raw(text) == expected


def test_negations_survive_stopword_removal():
    tokens = clean_tweet("this is not good at all", stem=False).split()
    assert "not" in tokens
    assert "is" not in tokens


def test_stemming():
    assert clean_tweet("loving running", remove_stopwords=False) == "love run"


def test_empty_and_symbol_only_input():
    assert clean_tweet("") == ""
    assert clean_tweet("!!! ??? ...") == ""


def test_transformer_matches_function():
    texts = ["I LOVE this!!", "worst day ever :("]
    assert TweetPreprocessor().fit_transform(texts) == [clean_tweet(t) for t in texts]
