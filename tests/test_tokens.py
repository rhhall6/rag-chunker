from rag_chunker.tokens import CHARS_PER_WORD_TOKEN, estimate_tokens, fits_budget


def test_empty_text_costs_nothing():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_ascii_words_are_roughly_four_characters_per_token():
    word = "x" * 21  # 21 characters -> ceil(21 / 4) = 6
    assert estimate_tokens(word) == 6


def test_short_word_still_costs_at_least_one_token():
    assert estimate_tokens("a") == 1
    assert estimate_tokens("of") == 1


def test_digit_runs_use_their_own_denser_rate():
    # "123456" is 6 digits -> ceil(6 / 3) = 2, cheaper than the word rate
    assert estimate_tokens("123456") == 2


def test_cjk_characters_cost_one_token_each():
    assert estimate_tokens("日本語") == 3


def test_newlines_and_symbols_are_cheap_but_not_free():
    assert estimate_tokens("\n\n\n") == 2  # ceil(3 * 0.5)
    assert estimate_tokens("...") == 2  # ceil(3 * 0.6)


def test_spaces_attach_to_the_following_token_for_free():
    assert estimate_tokens("hello") == estimate_tokens("   hello")


def test_mixed_text_sums_each_run():
    text = "abcd 123456 \n"
    expected = (
        max(1, 4 / CHARS_PER_WORD_TOKEN)  # "abcd"
        + max(1, 6 / 3.0)  # "123456"
        + 1 * 0.5  # "\n"
    )
    import math

    assert estimate_tokens(text) == int(math.ceil(expected))


def test_fits_budget_matches_estimate_tokens():
    text = "one two three four"
    estimate = estimate_tokens(text)
    assert fits_budget(text, estimate)
    assert not fits_budget(text, estimate - 1)
    assert fits_budget(text, estimate + 1)
