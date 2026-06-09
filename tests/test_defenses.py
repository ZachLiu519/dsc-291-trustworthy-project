from jbb_repro.defenses import dictionary_filter_prompt


def test_dictionary_filter_prompt_removes_non_dictionary_words() -> None:
    filtered = dictionary_filter_prompt("hello qzxqzx world!!!")

    assert filtered == "hello world!!!"
