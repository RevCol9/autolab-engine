from __future__ import annotations

import json
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from toolkit.model_crypto.artifact_monitor import (
    ArtifactMonitorConfig,
    LinuxInotifyArtifactMonitor,
    _decode_inotify_buffer,
    _is_forbidden_path,
    _normalize_suffixes,
    _write_json_atomic,
)


class ArtifactMonitorUnitTest(unittest.TestCase):
    def test_suffixes_are_normalized_and_temporary_forms_are_forbidden(self):
        suffixes = _normalize_suffixes(["PT", ".pth", "pt"])
        self.assertEqual((".pt", ".pth"), suffixes)
        self.assertTrue(_is_forbidden_path(Path("best.pt"), suffixes))
        self.assertTrue(_is_forbidden_path(Path(".best.pt.tmp"), suffixes))
        self.assertFalse(_is_forbidden_path(Path("best.niii-model"), suffixes))

    def test_decoder_rejects_truncated_event(self):
        with self.assertRaisesRegex(RuntimeError, "截断"):
            _decode_inotify_buffer(b"too short")

    def test_decoder_returns_all_event_fields(self):
        name = b"best.pt\x00"
        name += b"\x00" * ((4 - len(name) % 4) % 4)
        payload = struct.pack("=iIII", 7, 0x00000100, 23, len(name)) + name
        self.assertEqual(
            [(7, 0x00000100, 23, "best.pt")],
            _decode_inotify_buffer(payload),
        )

    def test_evidence_files_must_be_outside_watched_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "监控根目录之外"):
                ArtifactMonitorConfig.build(
                    roots=[root],
                    event_log=root / "events.jsonl",
                    summary=root.parent / "summary.json",
                )

    def test_stop_file_must_be_outside_watched_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "监控根目录之外"):
                ArtifactMonitorConfig.build(
                    roots=[root],
                    event_log=root.parent / "events.jsonl",
                    summary=root.parent / "summary.json",
                    stop_file=root / "stop",
                )

    def test_stale_stop_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "run"
            root.mkdir()
            stop = parent / "stop"
            stop.touch()
            with self.assertRaisesRegex(FileExistsError, "停止文件已存在"):
                ArtifactMonitorConfig.build(
                    roots=[root],
                    event_log=parent / "events.jsonl",
                    summary=parent / "summary.json",
                    stop_file=stop,
                )

    def test_control_and_evidence_paths_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "run"
            root.mkdir()
            shared = parent / "shared.json"
            with self.assertRaisesRegex(ValueError, "必须彼此不同"):
                ArtifactMonitorConfig.build(
                    roots=[root],
                    event_log=shared,
                    summary=parent / "summary.json",
                    ready_file=shared,
                )

    def test_atomic_json_publish_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "evidence.json"
            _write_json_atomic(target, {"session": "first"})

            with self.assertRaises(FileExistsError):
                _write_json_atomic(target, {"session": "second"})

            self.assertEqual({"session": "first"}, json.loads(target.read_text()))


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux inotify")
class LinuxInotifyArtifactMonitorIntegrationTest(unittest.TestCase):
    def _run_monitor(self, create_artifacts):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "run"
            evidence = parent / "evidence"
            root.mkdir()
            evidence.mkdir()
            ready = evidence / "ready.json"
            stop = evidence / "stop"
            events = evidence / "events.jsonl"
            summary = evidence / "summary.json"
            config = ArtifactMonitorConfig.build(
                roots=[root],
                event_log=events,
                summary=summary,
                ready_file=ready,
                stop_file=stop,
                poll_interval_sec=0.01,
            )
            monitor = LinuxInotifyArtifactMonitor(config)
            result_holder = {}

            def run():
                result_holder["result"] = monitor.run()

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.is_file(), "monitor did not become ready")
            create_artifacts(root)
            stop.touch()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "monitor did not stop")
            records = [json.loads(line) for line in events.read_text().splitlines()]
            return result_holder["result"], json.loads(summary.read_text()), records

    def test_clean_recursive_run_passes(self):
        def create(root):
            weights = root / "weights"
            weights.mkdir()
            time.sleep(0.05)
            (weights / "best.niii-model").write_bytes(b"ciphertext")

        result, summary, records = self._run_monitor(create)
        self.assertTrue(result.passed)
        self.assertEqual("passed", summary["status"])
        self.assertEqual(0, summary["forbiddenEventCount"])
        self.assertTrue(any(record["kind"] == "filesystem_event" for record in records))
        self.assertTrue(
            any(
                record.get("path", "").endswith("best.niii-model")
                for record in records
            )
        )

    def test_transient_plaintext_file_fails_even_after_deletion(self):
        def create(root):
            plaintext = root / "best.pt"
            plaintext.write_bytes(b"plaintext")
            plaintext.unlink()

        result, summary, records = self._run_monitor(create)
        self.assertFalse(result.passed)
        self.assertGreater(summary["forbiddenEventCount"], 0)
        self.assertTrue(
            any(
                record["kind"] == "forbidden_artifact"
                and record["path"].endswith("best.pt")
                for record in records
            )
        )


if __name__ == "__main__":
    unittest.main()
