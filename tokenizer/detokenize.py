from dataclasses import dataclass
from typing import Dict, List

from transformers import PreTrainedTokenizerBase

from ..message.tokenize import DeTokenizeMsg


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    read_offset: int


class DetokenizeManager:
    """Incrementally decode Qwen ByteLevel tokens without text cleanup.

    Replacement characters defer the entire pending fragment until more tokens
    arrive or the request finishes. This is a heuristic, not UTF-8 validation.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def discard(self, uid: int) -> None:
        """Release a cancelled or failed request without emitting pending text."""
        self.decode_map.pop(uid, None)

    def detokenize(self, msgs: List[DeTokenizeMsg]) -> List[str]:
        outputs: List[str] = []
        batch: List[DeTokenizeMsg] = []
        seen: set[int] = set()
        try:
            for msg in msgs:
                # A repeated UID must see the preceding token's committed offset.
                if msg.uid in seen:
                    outputs.extend(self._decode_batch(batch))
                    batch = []
                    seen.clear()
                batch.append(msg)
                seen.add(msg.uid)
            outputs.extend(self._decode_batch(batch))
        except Exception:
            # No outputs are returned on failure; callers must terminate these
            # requests rather than continue from partially committed state.
            for msg in msgs:
                self.discard(msg.uid)
            raise
        return outputs

    def _decode_batch(self, msgs: List[DeTokenizeMsg]) -> List[str]:
        if not msgs:
            return []

        pending_ids: List[List[int]] = []
        for msg in msgs:
            status = self.decode_map.setdefault(msg.uid, DecodeStatus([], 0))
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.decoded_ids.append(msg.next_token)
            pending_ids.append(status.decoded_ids[status.read_offset :])

        texts = self.tokenizer.batch_decode(
            pending_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        outputs: List[str] = []
        for msg, text in zip(msgs, texts, strict=True):
            status = self.decode_map[msg.uid]
            if msg.finished:
                outputs.append(text)
                self.discard(msg.uid)
            elif "\ufffd" not in text:
                outputs.append(text)
                status.read_offset = len(status.decoded_ids)
            else:
                outputs.append("")
        return outputs
