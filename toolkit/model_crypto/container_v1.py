"""Authenticated, pickle-free .niii-model container (first delivery format)."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"NIIIMODL"
FORMAT_VERSION = 1
MAX_PLAINTEXT_BYTES = 1024**3  # Whole-file AEAD: deliberately limit peak memory.
MAX_METADATA_BYTES = 16 * 1024**2
MAX_TENSORS = 100_000
MAX_KEY_ID_BYTES = 128
_HEADER = struct.Struct(">8sBBHQQ12s12s48s")
_WRAP_CONTEXT = b"niii-model-container-v1-dek\x00"
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool", "uint8", "int8", "int16", "int32", "int64",
        "float16", "bfloat16", "float32", "float64",
        "complex64", "complex128",
    )
}


@dataclass(frozen=True)
class ModelContainer:
    task: str
    model_yaml: dict[str, Any]
    model_names: dict[int, str]
    state_dict: dict[str, torch.Tensor]
    key_id: str


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"模型元数据包含不支持的类型: {type(value).__name__}")


def _encode_payload(
    state_dict: Mapping[str, torch.Tensor],
    model_yaml: Mapping[str, Any],
    model_names: Mapping[int | str, str],
    task: str,
) -> bytes:
    if task not in {"detect", "segment"}:
        raise ValueError("task 仅支持 detect 或 segment")
    if not state_dict or len(state_dict) > MAX_TENSORS:
        raise ValueError("state_dict 为空或张量数量超限")
    if any(not isinstance(name, str) or not name for name in state_dict):
        raise ValueError("state_dict 张量名称必须是非空字符串")
    normalized_names: dict[str, str] = {}
    for key, value in model_names.items():
        if isinstance(key, bool) or not str(key).isdigit() or not str(value):
            raise ValueError("model_names 必须是非负类别 ID 到非空名称的映射")
        normalized_key = str(int(key))
        if normalized_key in normalized_names:
            raise ValueError(f"model_names 存在重复类别 ID: {key}")
        normalized_names[normalized_key] = str(value)
    if not normalized_names:
        raise ValueError("model_names 不能为空")

    entries: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError(f"不支持的张量布局: {name}")
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if dtype_name not in _DTYPES:
            raise TypeError(f"不支持的张量 dtype: {name}={tensor.dtype}")
        raw = (
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            if tensor.numel() else b""
        )
        entries.append({
            "name": name,
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "offset": offset,
            "length": len(raw),
        })
        chunks.append(raw)
        offset += len(raw)
        if offset > MAX_PLAINTEXT_BYTES:
            raise ValueError("模型超出首版容器的 1 GiB 内存加密上限")

    metadata = {
        "task": task,
        "byte_order": sys.byteorder,
        "model_yaml": _json_value(dict(model_yaml)),
        "model_names": normalized_names,
        "tensors": entries,
    }
    encoded = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("模型元数据超出首版容器的 16 MiB 上限")
    if 4 + len(encoded) + offset > MAX_PLAINTEXT_BYTES:
        raise ValueError("模型超出首版容器的 1 GiB 内存加密上限")
    return struct.pack(">I", len(encoded)) + encoded + b"".join(chunks)


def write_model_container(
    destination: str | Path,
    *,
    state_dict: Mapping[str, torch.Tensor],
    model_yaml: Mapping[str, Any],
    model_names: Mapping[int | str, str],
    task: str,
    kek: bytes,
    key_id: str,
) -> Path:
    """Write only ciphertext to disk; source tensors stay in process memory."""
    destination = Path(destination)
    if destination.suffix != ".niii-model":
        raise ValueError("加密模型目标文件必须以 .niii-model 结尾")
    if destination.exists():
        raise FileExistsError(f"加密模型已存在，请使用新的工件 ID: {destination}")
    if len(kek) != 32:
        raise ValueError("KEK 必须为 32 字节")
    key_id_bytes = key_id.encode("utf-8")
    if not 0 < len(key_id_bytes) <= MAX_KEY_ID_BYTES:
        raise ValueError("key_id 长度必须在 1..128 字节之间")

    plaintext = _encode_payload(state_dict, model_yaml, model_names, task)
    dek = os.urandom(32)
    wrap_nonce = os.urandom(12)
    payload_nonce = os.urandom(12)
    wrapped_dek = AESGCM(kek).encrypt(wrap_nonce, dek, _WRAP_CONTEXT + key_id_bytes)
    header = _HEADER.pack(
        MAGIC, FORMAT_VERSION, 0, len(key_id_bytes), len(plaintext) + 16,
        len(plaintext), wrap_nonce, payload_nonce, wrapped_dek,
    )
    ciphertext = AESGCM(dek).encrypt(payload_nonce, plaintext, header + key_id_bytes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(header)
            stream.write(key_id_bytes)
            stream.write(ciphertext)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _decode_payload(payload: bytes, key_id: str) -> ModelContainer:
    if len(payload) < 4:
        raise ValueError("加密模型载荷缺少元数据长度")
    metadata_length = struct.unpack_from(">I", payload)[0]
    if not 0 < metadata_length <= MAX_METADATA_BYTES or 4 + metadata_length > len(payload):
        raise ValueError("加密模型元数据长度非法")
    metadata = json.loads(payload[4 : 4 + metadata_length])
    if not isinstance(metadata, dict) or metadata.get("task") not in {"detect", "segment"}:
        raise ValueError("加密模型任务或元数据格式非法")
    if metadata.get("byte_order") != sys.byteorder:
        raise ValueError("加密模型字节序与当前服务器不兼容")
    if not isinstance(metadata.get("model_yaml"), dict):
        raise ValueError("加密模型缺少 model_yaml")
    names = metadata.get("model_names")
    entries = metadata.get("tensors")
    if not isinstance(names, dict) or not isinstance(entries, list):
        raise ValueError("加密模型类别或张量索引格式非法")
    if not entries or len(entries) > MAX_TENSORS:
        raise ValueError("加密模型张量数量非法")

    tensor_bytes = memoryview(payload)[4 + metadata_length :]
    state_dict: dict[str, torch.Tensor] = {}
    offset = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("加密模型张量索引格式非法")
        name = entry.get("name")
        dtype_name = entry.get("dtype")
        dtype = _DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        shape = entry.get("shape")
        length = entry.get("length")
        if not isinstance(name, str) or not name or name in state_dict or dtype is None:
            raise ValueError("加密模型张量名称或 dtype 非法")
        if not isinstance(shape, list) or len(shape) > 16 or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise ValueError(f"加密模型张量 shape 非法: {name}")
        if (
            type(length) is not int or length < 0
            or type(entry.get("offset")) is not int or entry["offset"] != offset
        ):
            raise ValueError(f"加密模型张量 offset/length 非法: {name}")
        count = 1
        for dim in shape:
            count *= dim
            if count > MAX_PLAINTEXT_BYTES:
                raise ValueError(f"加密模型张量 shape 超限: {name}")
        if length != count * torch.empty((), dtype=dtype).element_size():
            raise ValueError(f"加密模型张量字节长度与 dtype/shape 不匹配: {name}")
        if offset + length > len(tensor_bytes):
            raise ValueError(f"加密模型张量越界: {name}")
        if length:
            raw = bytearray(tensor_bytes[offset : offset + length])
            tensor = torch.frombuffer(raw, dtype=dtype).clone().reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=dtype)
        state_dict[name] = tensor
        offset += length
    if offset != len(tensor_bytes):
        raise ValueError("加密模型载荷含未引用的尾部字节")
    model_names: dict[int, str] = {}
    for key, value in names.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(value, str):
            raise ValueError("加密模型类别索引非法")
        class_id = int(key)
        if str(class_id) != key or class_id in model_names or not value:
            raise ValueError("加密模型类别索引重复或名称为空")
        model_names[class_id] = value
    if not model_names:
        raise ValueError("加密模型类别不能为空")
    return ModelContainer(
        task=metadata["task"], model_yaml=metadata["model_yaml"],
        model_names=model_names, state_dict=state_dict, key_id=key_id,
    )


def read_model_container(path: str | Path, *, kek: bytes) -> ModelContainer:
    """Authenticate the entire ciphertext before parsing metadata or tensor bytes."""
    if len(kek) != 32:
        raise ValueError("KEK 必须为 32 字节")
    path = Path(path)
    if path.suffix != ".niii-model":
        raise ValueError("模型文件必须以 .niii-model 结尾")
    max_file_bytes = _HEADER.size + MAX_KEY_ID_BYTES + MAX_PLAINTEXT_BYTES + 16
    if path.stat().st_size > max_file_bytes:
        raise ValueError("加密模型超过首版容器的 1 GiB 上限")
    with path.open("rb") as stream:
        data = stream.read(max_file_bytes + 1)
    if len(data) > max_file_bytes:
        raise ValueError("加密模型超过首版容器的 1 GiB 上限")
    if len(data) < _HEADER.size:
        raise ValueError("加密模型文件头不完整")
    (
        magic, version, flags, key_id_length, cipher_length, plain_length,
        wrap_nonce, payload_nonce, wrapped_dek,
    ) = _HEADER.unpack_from(data)
    if magic != MAGIC or version != FORMAT_VERSION or flags != 0:
        raise ValueError("不支持的加密模型文件头或版本")
    if not 0 < key_id_length <= MAX_KEY_ID_BYTES:
        raise ValueError("加密模型 key_id 长度非法")
    if cipher_length != plain_length + 16 or plain_length > MAX_PLAINTEXT_BYTES:
        raise ValueError("加密模型密文长度非法")
    if len(data) != _HEADER.size + key_id_length + cipher_length:
        raise ValueError("加密模型文件长度与文件头不一致")
    key_id_bytes = data[_HEADER.size : _HEADER.size + key_id_length]
    authenticated_header = data[: _HEADER.size + key_id_length]
    ciphertext = data[_HEADER.size + key_id_length :]
    try:
        dek = AESGCM(kek).decrypt(
            wrap_nonce, wrapped_dek, _WRAP_CONTEXT + key_id_bytes,
        )
        plaintext = AESGCM(dek).decrypt(
            payload_nonce, ciphertext, authenticated_header,
        )
    except InvalidTag as exc:
        raise ValueError("加密模型认证失败：密钥错误或文件已损坏") from exc
    if len(plaintext) != plain_length:
        raise ValueError("加密模型解密长度不匹配")
    return _decode_payload(plaintext, key_id_bytes.decode("utf-8"))


__all__ = [
    "FORMAT_VERSION", "MAGIC", "MAX_PLAINTEXT_BYTES", "ModelContainer",
    "read_model_container", "write_model_container",
]
