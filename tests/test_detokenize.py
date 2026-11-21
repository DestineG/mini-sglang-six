import unittest

from sglang.message.tokenize import DeTokenizeMsg
from sglang.tokenizer.detokenize import DetokenizeManager


class ByteTokenizer:
    """Small ByteLevel fixture with split characters and special tokens."""

    eos_token_id = 0
    pieces = {
        0: b"<eos>",
        1: b"\xe4",
        2: b"\xb8",
        3: b"\xad",
        4: b"hello",
        5: b"\xef\xbf\xbd",
        6: b"\xff",
        7: b"A",
        8: b"<special>",
        9: b"\xf0\x9f",
        10: b"\x98\x80",
    }

    def __init__(self):
        self.calls = []
        self.fail = False

    def batch_decode(self, sequences, **kwargs):
        if self.fail:
            raise RuntimeError("decode failed")
        self.calls.append([list(ids) for ids in sequences])
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return [
            b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")
            for ids in sequences
        ]


class DetokenizeTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()
        self.manager = DetokenizeManager(self.tokenizer)

    def send(self, token, finished=False, uid=1):
        return self.manager.detokenize([DeTokenizeMsg(uid, token, finished)])[0]

    def test_split_chinese_and_emoji(self):
        for ids, expected in [([1, 2, 3], "\u4e2d"), ([9, 10], "\U0001f600")]:
            for token in ids[:-1]:
                self.assertEqual(self.send(token), "")
            self.assertEqual(self.send(ids[-1]), expected)
        self.assertEqual(self.send(0, finished=True), "")
        self.assertFalse(self.manager.decode_map)

    def test_repeated_uid_in_batch_and_interleaved_requests(self):
        msgs = [
            DeTokenizeMsg(1, 1, False),
            DeTokenizeMsg(2, 4, False),
            DeTokenizeMsg(1, 2, False),
            DeTokenizeMsg(2, 7, True),
            DeTokenizeMsg(1, 3, True),
        ]
        self.assertEqual(self.manager.detokenize(msgs), ["", "hello", "", "A", "\u4e2d"])
        self.assertEqual(len(self.tokenizer.calls[0]), 2)
        self.assertFalse(self.manager.decode_map)

    def test_replacement_and_invalid_bytes_flush_at_finish(self):
        for token in [1, 5, 6]:
            self.assertEqual(self.send(token), "")
            self.assertEqual(self.send(7), "")
            self.assertEqual(self.send(0, finished=True), "\ufffdA")
            self.assertFalse(self.manager.decode_map)

    def test_last_normal_token_and_special_token(self):
        self.assertEqual(self.send(4), "hello")
        self.assertEqual(self.send(8), "<special>")
        self.assertEqual(self.send(7, finished=True), "A")
        self.assertFalse(self.manager.decode_map)

    def test_empty_and_eos_only(self):
        self.assertEqual(self.manager.detokenize([]), [])
        self.assertEqual(self.send(0, finished=True), "")
        self.assertFalse(self.manager.decode_map)

    def test_cancel_and_failure_cleanup(self):
        self.send(1)
        self.send(1, uid=2)
        self.manager.discard(1)
        self.manager.discard(1)
        self.assertNotIn(1, self.manager.decode_map)
        self.tokenizer.fail = True
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            self.send(2, uid=2)
        self.assertFalse(self.manager.decode_map)


if __name__ == "__main__":
    unittest.main()
