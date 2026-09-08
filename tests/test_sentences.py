from rag_chunker.sentences import split_sentences


def test_empty_and_blank_text_produce_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_splits_on_terminal_punctuation():
    assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]


def test_keeps_the_terminating_punctuation_with_the_sentence():
    sentences = split_sentences("Is this it? Yes!")
    assert sentences == ["Is this it?", "Yes!"]


def test_closing_quotes_and_brackets_stay_with_the_sentence():
    text = 'She said "Stop." Then she left.'
    assert split_sentences(text) == ['She said "Stop."', "Then she left."]


def test_abbreviations_do_not_end_a_sentence():
    text = "Dr. Chen met Prof. Ito for coffee. They left at noon."
    assert split_sentences(text) == [
        "Dr. Chen met Prof. Ito for coffee.",
        "They left at noon.",
    ]


def test_eg_and_ie_do_not_end_a_sentence():
    text = "Bring supplies, e.g. rope and a map. Pack light."
    assert split_sentences(text) == [
        "Bring supplies, e.g. rope and a map.",
        "Pack light.",
    ]


def test_middle_initial_does_not_end_a_sentence():
    text = "Ask J. Smith about the report. It is overdue."
    assert split_sentences(text) == [
        "Ask J. Smith about the report.",
        "It is overdue.",
    ]


def test_decimal_and_dotted_version_numbers_do_not_end_a_sentence():
    text = "See section 1.2.3 for details. Then continue."
    assert split_sentences(text) == [
        "See section 1.2.3 for details.",
        "Then continue.",
    ]


def test_numbered_list_items_are_not_split_at_the_marker():
    text = "1. First item. 2. Second item."
    assert split_sentences(text) == ["1. First item.", "2. Second item."]


def test_no_terminal_punctuation_returns_the_whole_text_as_one_sentence():
    assert split_sentences("no ending punctuation here") == [
        "no ending punctuation here"
    ]


def test_surrounding_whitespace_is_stripped_from_each_sentence():
    assert split_sentences("  One.   Two.  ") == ["One.", "Two."]
