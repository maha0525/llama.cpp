"""Exercise actual serializer I/O failures with a Linux LD_PRELOAD shim.

Run with the usual server test setup and a dynamically linked llama-server:
    pytest unit/test_slot_save_faults.py
    SLOT_SAVE_EXHAUSTIVE_TESTS=1 pytest unit/test_slot_save_faults.py

The default checks early, middle, and late writes plus every flush/close/rename.
The exhaustive run fails each observed call in turn and can take several minutes.
All fault settings are private to this test shim; production code has no test flags.
"""

import base64
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

import pytest

from utils import ServerPreset


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="LD_PRELOAD fault injection needs Linux")


def _image():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\x00" + b"\xff\x00\x00" * 32 for _ in range(32))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def test_slot_save_serializer_failures(tmp_path, monkeypatch):
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if not compiler:
        pytest.skip("A C compiler is required for the fault injection shim")
    if os.environ.get("DEBUG_EXTERNAL"):
        pytest.skip("The test must start its own server with LD_PRELOAD")

    shim = tmp_path / "slot_save_faults.so"
    source = Path(__file__).parents[1] / "fixtures" / "slot_save_faults.c"
    subprocess.run([compiler, "-std=c11", "-shared", "-fPIC", str(source), "-ldl", "-o", str(shim)], check=True)
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    marker = tmp_path / "fault"
    trace = tmp_path / "trace"
    monkeypatch.setenv("LD_PRELOAD", ":".join(filter(None, (os.environ.get("LD_PRELOAD"), str(shim)))))
    monkeypatch.setenv("SLOT_SAVE_FAULT_ROOT", str(saved_dir))
    monkeypatch.setenv("SLOT_SAVE_FAULT_MARKER", str(marker))
    monkeypatch.setenv("SLOT_SAVE_FAULT_TRACE", str(trace))
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")

    server = ServerPreset.tinygemma3()
    server.slot_save_path = str(saved_dir)
    server.n_threads = 2
    server.temperature = 0.0
    server_path = server.server_path or os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    probe = subprocess.run([server_path, "--version"], capture_output=True, text=True, timeout=60)
    if "ASan runtime does not come first" in probe.stderr:
        pytest.skip("LD_PRELOAD fault injection is incompatible with this ASan runtime link order")
    assert probe.returncode == 0, probe.stdout + probe.stderr
    server.start()
    assert server.process is not None
    if str(shim) not in Path(f"/proc/{server.process.pid}/maps").read_text():
        server.stop()
        pytest.skip("LD_PRELOAD shim was not loaded (for example, a fully static server)")
    prompt = {
        "id_slot": 1,
        "cache_prompt": True,
        "n_predict": 4,
        "temperature": 0.0,
        "prompt": {
            "prompt_string": "Describe this: <__media__>\n" + "Give a short description of the picture. " * 8,
            "multimodal_data": [_image()],
        },
    }
    response = server.make_request("POST", "/completion", data=prompt)
    assert response.status_code == 200, response.body

    filename = "last-good.bin"
    endpoint = "/slots/1?action=save"
    response = server.make_request("POST", endpoint, data={"filename": filename})
    assert response.status_code == 200, response.body
    n_saved = response.body["n_saved"]
    destination = saved_dir / filename
    previous = destination.read_bytes()
    assert previous[:8] == b"LLSLOT\x00\x00"
    assert struct.unpack_from("<I", previous, 12)[0] == 7, "The save must contain real KV, media, and checkpoint payloads"
    assert all(struct.unpack_from("<QQQ", previous, 16))

    # Save a different generation to detect an accidental replacement of the old file.
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    prompt["prompt"]["prompt_string"] += "Please mention its main color."
    response = server.make_request("POST", "/completion", data=prompt)
    assert response.status_code == 200, response.body
    trace.write_text("")
    response = server.make_request("POST", endpoint, data={"filename": "probe.bin"})
    assert response.status_code == 200, response.body
    probe = saved_dir / "probe.bin"
    assert probe.read_bytes() != previous
    probe.unlink()

    # Record the real library calls instead of assuming how the C++ stream flushes.
    operations = [tuple(line.split()[:2]) for line in trace.read_text().splitlines()]
    observed = set(operations)
    for component in ("0", "2", "container"):
        assert {(op, component) for op in ("fwrite", "fflush", "fclose")} <= observed
    assert ("rename", "container") in observed
    assert any((op, "1") in observed for op in ("write", "writev", "fwrite"))
    assert any((op, "1") in observed for op in ("close", "fclose"))

    # Early, middle, and late writes cover headers, payloads, and footers.
    faults = []
    for op, component in sorted(observed):
        count = operations.count((op, component))
        occurrences = range(1, count + 1) if os.environ.get("SLOT_SAVE_EXHAUSTIVE_TESTS") == "1" else sorted({1, (count + 1) // 2, count})
        faults.extend((op, component, occurrence) for occurrence in occurrences)

    for generation, (op, component, occurrence) in enumerate(faults, 1):
        trace.write_text("")
        marker.write_text(f"{generation} {op} {component} {occurrence}\n")
        response = server.make_request("POST", endpoint, data={"filename": filename})
        marker.unlink()
        injected = f"{op} {component} 1" in trace.read_text().splitlines()
        assert injected, (op, component, occurrence, trace.read_text())
        assert response.status_code == 500, (op, component, occurrence, response.body)
        assert response.body["error"]["type"] == "server_error"
        assert destination.read_bytes() == previous, (op, component, occurrence)
        assert sorted(p.name for p in saved_dir.iterdir()) == [filename]

        response = server.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert response.status_code == 200, (op, component, occurrence, response.body)
        assert response.body["n_restored"] == n_saved
        assert destination.read_bytes() == previous

    server.stop()
    server.start()
    response = server.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
    assert response.status_code == 200, response.body
    assert response.body["n_restored"] == n_saved
    assert destination.read_bytes() == previous
    server.stop()
    assert sorted(p.name for p in saved_dir.iterdir()) == [filename]
    print(f"Verified {len(faults)} actual serializer/container faults across {sorted(observed)}")
