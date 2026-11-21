from typing import Dict, Any

import torch
import numpy as np


MESSAGE_REGISTRY = {}
def register_message(cls):
    message_type = cls.message_type

    if message_type in MESSAGE_REGISTRY:
        raise ValueError(f"Duplicate message type: {message_type}")

    MESSAGE_REGISTRY[message_type] = cls
    return cls

def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)

def serialize_type(obj) -> Dict:
    serialized = {}

    if isinstance(obj, torch.Tensor):
        assert obj.dim() == 1, "we can only serialize 1D tensor for now"

        serialized["__type__"] = "Tensor"
        serialized["buffer"] = obj.numpy().tobytes()
        serialized["dtype"] = str(obj.dtype)
        return serialized

    serialized["__type__"] = obj.message_type

    for k, v in obj.__dict__.items():
        serialized[k] = _serialize_any(v)

    return serialized

def _deserialize_any(data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(data)
        else:
            return {k: _deserialize_any(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")

def deserialize_type(data: Dict) -> Any:
    type_name = data["__type__"]

    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)

        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        return torch.from_numpy(np_tensor.copy())

    cls = MESSAGE_REGISTRY[type_name]

    kwargs = {}

    for k, v in data.items():
        if k == "__type__":
            continue

        kwargs[k] = _deserialize_any(v)

    return cls(**kwargs)
