"""
`apply_chat_template` return-type handling.

WHY THIS FILE EXISTS
--------------------
Job 11599044 burned a GPU allocation because `apply_chat_template` returned a
STRING on transformers 5.x and `list(a_string)` is a list of CHARACTERS. That
list is perfectly valid Python, so it flowed through request creation, through
admission, into the scheduler, and died at `torch.tensor(...)` with:

    ValueError: too many dimensions 'str'

an error that names neither the tokenizer nor the chat template. Every request
returned HTTP 200 with a well-formed, empty SSE stream.

These tests pin every return shape the function is known to take.
"""

import pytest

from serving.server.app import HFChatTokenizer

MSGS = [{"role": "user", "content": "hi"}]
IDS = [128000, 9906, 11]


class FakeTok:
    def __init__(self, ret):
        self._ret = ret
        self.called_with = None

    def apply_chat_template(self, messages, **kw):
        self.called_with = kw
        return self._ret

    def decode(self, ids, **kw):
        return "x"


class FakeTensor:
    def __init__(self, data):
        self._d = data

    def tolist(self):
        return self._d


def test_plain_list_of_ints():
    assert HFChatTokenizer(FakeTok(IDS)).encode_chat(MSGS) == IDS


def test_batched_nested_list():
    """Some versions return [[ids]] for a single conversation."""
    assert HFChatTokenizer(FakeTok([IDS])).encode_chat(MSGS) == IDS


def test_batch_encoding_dict():
    assert HFChatTokenizer(FakeTok({"input_ids": IDS})).encode_chat(MSGS) == IDS


def test_tensor_like():
    assert HFChatTokenizer(FakeTok(FakeTensor(IDS))).encode_chat(MSGS) == IDS


def test_nested_tensor_like():
    assert HFChatTokenizer(FakeTok(FakeTensor([IDS]))).encode_chat(MSGS) == IDS


def test_string_return_raises_immediately():
    """
    THE REGRESSION. A string must fail HERE, naming the cause, not 300 lines
    later inside torch as 'too many dimensions str'.
    """
    tok = HFChatTokenizer(FakeTok("<|begin_of_text|>hi<|eot_id|>"))
    with pytest.raises(TypeError, match="returned a string despite tokenize=True"):
        tok.encode_chat(MSGS)


def test_list_of_characters_raises():
    """
    The exact shape the old `list(apply_chat_template(...))` produced. It is a
    valid list, so only a type check catches it.
    """
    tok = HFChatTokenizer(FakeTok(list("<|begin")))
    with pytest.raises(TypeError, match="non-empty list"):
        tok.encode_chat(MSGS)


def test_empty_raises():
    with pytest.raises(TypeError, match="non-empty list"):
        HFChatTokenizer(FakeTok([])).encode_chat(MSGS)


def test_tokenize_true_is_passed_explicitly():
    """Do not rely on the default; it has changed across versions."""
    tok = FakeTok(IDS)
    HFChatTokenizer(tok).encode_chat(MSGS)
    assert tok.called_with.get("tokenize") is True
    assert tok.called_with.get("add_generation_prompt") is True
