from typing import Dict, Type, Any

import torch
import numpy as np


MESSAGE_REGISTRY = {}
def register_message(cls):
    message_type = cls.message_type

    if message_type in MESSAGE_REGISTRY:
        raise ValueError(f"Duplicate message type: {message_type}")

    MESSAGE_REGISTRY[message_type] = cls
    return cls

def serialize_type(self) -> Dict:
    ...

def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    ...