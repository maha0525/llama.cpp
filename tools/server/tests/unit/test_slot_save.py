import base64
import hashlib
import struct
import zlib
from pathlib import Path

import pytest
from utils import *

server = ServerPreset.tinyllama2()

SLOT_HEADER = struct.Struct("<8sIIQQQ")
SLOT_MAGIC = b"LLSLOT\0\0"
SLOT_SECTIONS = ("kv", "mtmd", "ckpt")


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0


def _read_container(data):
    magic, version, flags, *sizes = SLOT_HEADER.unpack_from(data)
    assert magic == SLOT_MAGIC
    assert version == 1
    assert flags & 1
    assert not flags & ~7
    assert len(data) == SLOT_HEADER.size + sum(sizes) + 32
    assert hashlib.sha256(data[:-32]).digest() == data[-32:]
    sections = {}
    offset = SLOT_HEADER.size
    for index, (name, size) in enumerate(zip(SLOT_SECTIONS, sizes)):
        assert flags & (1 << index) or size == 0
        sections[name] = data[offset:offset + size] if flags & (1 << index) else None
        offset += size
    return sections


def _make_container(kv, mtmd=None, ckpt=None):
    sections = (kv, mtmd, ckpt)
    flags = sum(1 << index for index, data in enumerate(sections) if data is not None)
    header = SLOT_HEADER.pack(SLOT_MAGIC, 1, flags, *(len(data) if data is not None else 0 for data in sections))
    data = header + b"".join(data for data in sections if data is not None)
    return data + hashlib.sha256(data).digest()


def _kv_tokens(kv):
    # The embedded llama_state_seq file uses native-endian 32-bit tokens.
    count = struct.unpack_from("=I", kv, 8)[0]
    return struct.unpack_from(f"={count}i", kv, 12)


def _save_slot(server, filename, id_slot=1):
    res = server.make_request("POST", f"/slots/{id_slot}?action=save", data={"filename": filename})
    assert res.status_code == 200
    data = (Path(server.slot_save_path) / filename).read_bytes()
    sections = _read_container(data)
    assert res.body["n_written"] == len(data)
    assert res.body["n_saved"] == len(_kv_tokens(sections["kv"]))
    return res.body, data, sections


def _restore_slot(server, filename, saved, id_slot=1):
    res = server.make_request("POST", f"/slots/{id_slot}?action=restore", data={"filename": filename})
    assert res.status_code == 200
    assert res.body["n_restored"] == saved["n_saved"]
    assert res.body["n_read"] == (Path(server.slot_save_path) / filename).stat().st_size


def _assert_rejected_without_mutation(server, data, before):
    path = Path(server.slot_save_path) / "invalid.bin"
    path.write_bytes(data)
    files_before = set(path.parent.iterdir())
    res = server.make_request("POST", "/slots/1?action=restore", data={"filename": path.name})
    assert res.status_code == 400
    assert set(path.parent.iterdir()) == files_before
    _, _, after = _save_slot(server, "after_rejected_restore.bin")
    assert after == before


def test_slot_save_restore():
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21

    saved, _, sections = _save_slot(server, "slot1.bin")
    assert saved["n_saved"] == 84
    assert sections["mtmd"] is None

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6

    _restore_slot(server, "slot1.bin", saved, id_slot=0)
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_save_file_error():
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "A short text prompt.", "id_slot": 1, "cache_prompt": True,
    })
    assert res.status_code == 200
    path = Path(server.slot_save_path) / "slot_save_directory.bin"
    path.mkdir()
    marker = path / "unrelated.txt"
    marker.write_bytes(b"keep this file")
    files_before = set(path.parent.iterdir())
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": path.name})
    assert res.status_code == 500
    assert res.body["error"]["type"] == "server_error"
    assert set(path.parent.iterdir()) == files_before
    assert marker.read_bytes() == b"keep this file"


def test_slot_restore_rejects_corrupt_and_legacy_files():
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?", "id_slot": 1, "cache_prompt": True,
    })
    assert res.status_code == 200
    _, good, before = _save_slot(server, "good.bin")
    corrupted = bytearray(good)
    corrupted[SLOT_HEADER.size + len(before["kv"]) // 2] ^= 1
    unknown_version = bytearray(good)
    struct.pack_into("<I", unknown_version, 8, 2)
    unknown_version[-32:] = hashlib.sha256(unknown_version[:-32]).digest()

    for bad in (bytes(corrupted), good[:SLOT_HEADER.size - 1], good[:-1], good + b"trailing", bytes(unknown_version), before["kv"]):
        _assert_rejected_without_mutation(server, bad, before)

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?", "id_slot": 1, "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6
    assert match_regex("(Jack|said)+", res.body["content"])


def test_slot_erase():
    server.start()
    prompt = {"prompt": "What is the capital of France?", "id_slot": 1, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21
    saved, data, _ = _save_slot(server, "erase.bin")

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200
    assert res.body["n_erased"] == saved["n_saved"]
    assert (Path(server.slot_save_path) / "erase.bin").read_bytes() == data

    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21


def _image_base64(rgb):
    # Equal-sized images give equal media-token layouts without a network fixture.
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    pixels = (b"\0" + bytes(rgb) * 32) * 32
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def _image_prompt(rgb=(255, 0, 0), id_slot=1, n_predict=4):
    return {
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": id_slot,
        "n_predict": n_predict,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_image_base64(rgb)],
        },
    }


@pytest.fixture
def mmproj_server(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")
    monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "8")
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = str(tmp_path)
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()
    prompt = {"prompt": "The quick brown fox jumps over the lazy dog.", "id_slot": 1, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] > 0
    content = res.body["content"]
    saved, _, sections = _save_slot(server, "mm_text.bin")
    assert saved["n_saved"] > 0
    assert sections["mtmd"] is None

    server.stop()
    server.start()
    _restore_slot(server, "mm_text.bin", saved, id_slot=0)
    res = server.make_request("POST", "/completion", data={**prompt, "id_slot": 0})
    assert res.status_code == 200
    assert res.body["content"] == content


def test_slot_save_restore_image(mmproj_server):
    server = mmproj_server
    server.start()
    prompt = _image_prompt()
    res = server.make_request("POST", "/completions", data=prompt)
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n = res.body["timings"]["prompt_n"]
    filename = "mm_image.bin"
    path = Path(server.slot_save_path) / filename
    saved, data, sections = _save_slot(server, filename)
    assert saved["n_saved"] > 0
    assert sections["mtmd"]
    assert sections["ckpt"]
    assert set(path.parent.iterdir()) == {path}

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200
    assert res.body["n_erased"] == saved["n_saved"]
    assert path.read_bytes() == data

    # A fresh process has no active task when it restores the saved slot.
    server.stop()
    server.start()
    _restore_slot(server, filename, saved)
    res = server.make_request("POST", "/completions", data=prompt)
    assert res.status_code == 200
    assert res.body["content"] == content
    assert res.body["timings"]["prompt_n"] < prompt_n

    # An absent checkpoint is valid and falls back to prompt recomputation.
    server.stop()
    path.write_bytes(_make_container(sections["kv"], sections["mtmd"]))
    server.start()
    _restore_slot(server, filename, saved)
    res = server.make_request("POST", "/completions", data=prompt)
    assert res.status_code == 200
    assert res.body["content"] == content
    assert res.body["timings"]["prompt_n"] == prompt_n


def test_slot_save_replaces_media_and_checkpoints(mmproj_server, monkeypatch):
    server = mmproj_server
    server.start()
    res = server.make_request("POST", "/completions", data=_image_prompt())
    assert res.status_code == 200
    filename = "reused.bin"
    saved, _, sections = _save_slot(server, filename)
    assert sections["mtmd"]
    assert sections["ckpt"]

    # Files from older servers are external to the container and must be ignored.
    path = Path(server.slot_save_path) / filename
    unrelated = {
        Path(str(path) + ".mtmd"): b"stale external media",
        Path(str(path) + ".ckpt"): b"stale external checkpoint",
        path.parent / "keep.txt": b"unrelated file",
    }
    for external, data in unrelated.items():
        external.write_bytes(data)
    _restore_slot(server, filename, saved, id_slot=0)

    server.stop()
    monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "0")
    server.start()
    prompt = {"prompt": "A short text prompt.", "id_slot": 1, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    content = res.body["content"]
    saved, _, sections = _save_slot(server, filename)
    assert sections["mtmd"] is None
    assert sections["ckpt"] is None
    assert set(path.parent.iterdir()) == {path, *unrelated}
    for external, data in unrelated.items():
        assert external.read_bytes() == data

    server.stop()
    server.start()
    _restore_slot(server, filename, saved)
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    assert res.body["content"] == content
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    for external, data in unrelated.items():
        assert external.read_bytes() == data
    assert path.is_file()


def test_slot_restore_text_replaces_occupied_image_slot(mmproj_server):
    server = mmproj_server
    server.start()
    prompt = {"prompt": "A short text prompt.", "id_slot": 0, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n = res.body["timings"]["prompt_n"]
    saved, _, text = _save_slot(server, "text.bin", id_slot=0)
    (Path(server.slot_save_path) / "text.bin").write_bytes(_make_container(text["kv"]))

    res = server.make_request("POST", "/completions", data=_image_prompt())
    assert res.status_code == 200
    _, _, occupied = _save_slot(server, "occupied.bin")
    assert occupied["mtmd"]
    assert occupied["ckpt"]
    assert len(_kv_tokens(occupied["kv"])) > saved["n_saved"]

    _restore_slot(server, "text.bin", saved)
    _, _, restored = _save_slot(server, "restored.bin")
    assert restored["mtmd"] is None
    assert _kv_tokens(restored["kv"]) == _kv_tokens(text["kv"])
    assert restored["ckpt"]
    assert restored["ckpt"] != occupied["ckpt"]
    assert struct.unpack_from("=I", restored["ckpt"], 8)[0] == 1
    assert struct.unpack_from("=q", restored["ckpt"], 12)[0] == saved["n_saved"]

    res = server.make_request("POST", "/completion", data={**prompt, "id_slot": 1})
    assert res.status_code == 200
    assert res.body["content"] == content
    assert res.body["timings"]["prompt_n"] == prompt_n


def test_slot_restore_rejects_trailing_kv_and_clears_slot(mmproj_server):
    server = mmproj_server
    server.start()
    prompt = _image_prompt()
    res = server.make_request("POST", "/completions", data=prompt)
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n = res.body["timings"]["prompt_n"]
    _, _, sections = _save_slot(server, "valid.bin")
    assert sections["mtmd"]
    assert sections["ckpt"]
    sections["kv"] += b"invalid trailing state bytes"
    path = Path(server.slot_save_path) / "invalid_kv.bin"
    path.write_bytes(_make_container(**sections))

    # The binding is valid; the low-level state loader must reject the unused KV bytes.
    res = server.make_request("POST", "/slots/1?action=restore", data={"filename": path.name})
    assert res.status_code == 400
    saved, _, cleared = _save_slot(server, "cleared.bin")
    assert saved["n_saved"] == 0
    assert cleared["mtmd"] is None
    assert cleared["ckpt"] is None
    res = server.make_request("POST", "/completions", data=prompt)
    assert res.status_code == 200
    assert res.body["content"] == content
    assert res.body["timings"]["prompt_n"] == prompt_n


def test_slot_restore_rejects_invalid_media_and_checkpoints(mmproj_server):
    server = mmproj_server
    server.start()
    res = server.make_request("POST", "/completions", data=_image_prompt())
    assert res.status_code == 200
    _, _, image = _save_slot(server, "image.bin")
    assert image["mtmd"]
    assert image["ckpt"]

    # The destination has a different usable prompt before each rejected restore.
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    prompt = {"prompt": "Keep this text prompt intact.", "id_slot": 1, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    content = res.body["content"]
    _, _, before = _save_slot(server, "before.bin")

    wrong_hash = bytearray(image["mtmd"])
    wrong_hash[8] ^= 1
    invalid_checkpoint = bytearray(image["ckpt"])
    struct.pack_into("=I", invalid_checkpoint, 8, 1025)
    invalid_checkpoint_tokens = bytearray(image["ckpt"])
    struct.pack_into("=q", invalid_checkpoint_tokens, 12, len(_kv_tokens(image["kv"])) + 1)
    invalid_sections = (
        {**image, "mtmd": bytes(wrong_hash)},
        {**image, "mtmd": image["mtmd"][:-1]},
        {**image, "mtmd": None},
        {**image, "ckpt": bytes(invalid_checkpoint)},
        {**image, "ckpt": bytes(invalid_checkpoint_tokens)},
        {**image, "ckpt": image["ckpt"][:-1]},
        {**image, "ckpt": b""},
    )
    for sections in invalid_sections:
        # Recompute the binding so rejection must come from section validation.
        _assert_rejected_without_mutation(server, _make_container(**sections), before)

    # An old KV plus valid external sidecars still has no binding and is rejected.
    path = Path(server.slot_save_path) / "invalid.bin"
    Path(str(path) + ".mtmd").write_bytes(image["mtmd"])
    Path(str(path) + ".ckpt").write_bytes(image["ckpt"])
    _assert_rejected_without_mutation(server, image["kv"], before)
    assert Path(str(path) + ".mtmd").read_bytes() == image["mtmd"]
    assert Path(str(path) + ".ckpt").read_bytes() == image["ckpt"]
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    assert res.body["content"] == content


def test_slot_restore_rejects_mixed_media_generations(mmproj_server):
    server = mmproj_server
    server.start()
    for index, rgb in enumerate(((255, 0, 0), (0, 0, 255))):
        res = server.make_request("POST", "/completions", data=_image_prompt(rgb, id_slot=index, n_predict=1))
        assert res.status_code == 200
    _, red_data, red = _save_slot(server, "red.bin", id_slot=0)
    _, _, blue = _save_slot(server, "blue.bin")
    assert _kv_tokens(red["kv"]) == _kv_tokens(blue["kv"])
    assert red["mtmd"] != blue["mtmd"]
    assert red["ckpt"] != blue["ckpt"]

    for section in ("kv", "mtmd", "ckpt"):
        mixed = _make_container(**{**red, section: blue[section]})
        # Keep the original generation's binding, even if section lengths differ.
        mixed = mixed[:-32] + red_data[-32:]
        _assert_rejected_without_mutation(server, mixed, blue)


def test_slot_erase_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()
    prompt = {"prompt": "The quick brown fox jumps over the lazy dog.", "id_slot": 1, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    res = server.make_request("POST", "/completion", data=prompt)
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == prompt_n
