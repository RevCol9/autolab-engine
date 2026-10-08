from __future__ import annotations

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.rotate_model_key import rotate_model_key
from toolkit.model_crypto.container_v1 import (
    read_model_container,
    read_model_key_id,
    write_model_container,
)
from toolkit.model_crypto.envelope import (
    init_kek,
    keyring_status,
    load_kek,
    rotate_kek,
    set_active_kek,
)


class LocalKeyringTest(unittest.TestCase):
    def test_active_key_can_rotate_and_roll_back_without_losing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / "keys"
            old_path, old_id = init_kek(keys, kek_id="key-2026-a")
            old_material = old_path.read_bytes()

            new_path, new_id = rotate_kek(keys, kek_id="key-2026-b")
            new_material, active_id = load_kek(key_dir=keys)

            self.assertEqual("key-2026-b", new_id)
            self.assertEqual(new_id, active_id)
            self.assertEqual(new_path.read_bytes(), new_material)
            self.assertNotEqual(old_material, new_material)
            self.assertEqual(
                (old_material, old_id),
                load_kek(key_dir=keys, key_id=old_id),
            )
            self.assertEqual(
                ("key-2026-b", ("key-2026-a", "key-2026-b")),
                keyring_status(keys),
            )

            set_active_kek(keys, old_id)
            self.assertEqual((old_material, old_id), load_kek(key_dir=keys))
            self.assertEqual(
                ("key-2026-a", ("key-2026-a", "key-2026-b")),
                keyring_status(keys),
            )

    def test_first_rotation_migrates_the_legacy_single_key_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / "keys"
            keys.mkdir()
            legacy_material = os.urandom(32)
            legacy_path = keys / "kek.key"
            legacy_path.write_bytes(legacy_material)
            (keys / "kek_id.txt").write_text("legacy\n", encoding="utf-8")
            if os.name != "nt":
                legacy_path.chmod(0o600)

            rotate_kek(keys, kek_id="current")

            self.assertEqual(
                (legacy_material, "legacy"),
                load_kek(key_dir=keys, key_id="legacy"),
            )
            self.assertEqual("current", load_kek(key_dir=keys)[1])
            self.assertEqual(
                ("current", ("current", "legacy")),
                keyring_status(keys),
            )

    def test_failed_key_write_does_not_leave_a_partial_history_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / "keys"
            init_kek(keys, kek_id="old")

            with patch(
                "toolkit.model_crypto.envelope.os.fsync",
                side_effect=OSError(errno.ENOSPC, "injected disk full"),
            ):
                with self.assertRaises(OSError):
                    rotate_kek(keys, kek_id="partial")

            self.assertEqual(("old", ("old",)), keyring_status(keys))

    def test_file_keyring_commands_ignore_runtime_key_id_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / "keys"
            init_kek(keys, kek_id="old")

            with patch.dict(os.environ, {"NIII_MODEL_KEK_ID": "runtime-only"}):
                rotate_kek(keys, kek_id="new")
                self.assertEqual(("new", ("new", "old")), keyring_status(keys))

    def test_active_pointer_failure_keeps_old_active_and_complete_new_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / "keys"
            init_kek(keys, kek_id="old")

            with patch(
                "toolkit.model_crypto.envelope.os.replace",
                side_effect=OSError(errno.EIO, "injected pointer failure"),
            ):
                with self.assertRaises(OSError):
                    rotate_kek(keys, kek_id="new")

            self.assertEqual(("old", ("new", "old")), keyring_status(keys))


class ModelKeyRotationTest(unittest.TestCase):
    def test_rotation_writes_a_new_authenticated_artifact_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "keys"
            old_kek_path, old_id = init_kek(keys, kek_id="old")
            source = root / "model-old.niii-model"
            state_dict = {
                "weight": torch.tensor([1.0, 2.0], dtype=torch.float16),
                "counter": torch.tensor([2**53 + 1], dtype=torch.int64),
            }
            write_model_container(
                source,
                state_dict=state_dict,
                model_yaml={"nc": 1},
                model_names={0: "item"},
                task="detect",
                kek=old_kek_path.read_bytes(),
                key_id=old_id,
            )
            source_bytes = source.read_bytes()
            rotate_kek(keys, kek_id="new")
            destination = root / "model-new.niii-model"

            result = rotate_model_key(source, destination, key_dir=keys)

            self.assertEqual(destination, result)
            self.assertEqual(source_bytes, source.read_bytes())
            self.assertEqual("old", read_model_key_id(source))
            self.assertEqual("new", read_model_key_id(destination))
            old_container = read_model_container(
                source,
                kek=load_kek(key_dir=keys, key_id="old")[0],
            )
            new_container = read_model_container(
                destination,
                kek=load_kek(key_dir=keys, key_id="new")[0],
            )
            self.assertEqual(old_container.task, new_container.task)
            self.assertEqual(old_container.model_yaml, new_container.model_yaml)
            self.assertEqual(old_container.model_names, new_container.model_names)
            self.assertEqual(set(old_container.state_dict), set(new_container.state_dict))
            for name in old_container.state_dict:
                self.assertEqual(
                    old_container.state_dict[name].dtype,
                    new_container.state_dict[name].dtype,
                )
                self.assertTrue(
                    torch.equal(
                        old_container.state_dict[name].view(torch.uint8),
                        new_container.state_dict[name].view(torch.uint8),
                    )
                )

    def test_rotation_never_overwrites_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "keys"
            old_path, old_id = init_kek(keys, kek_id="old")
            source = root / "source.niii-model"
            destination = root / "destination.niii-model"
            for path in (source, destination):
                write_model_container(
                    path,
                    state_dict={"weight": torch.ones(1)},
                    model_yaml={"nc": 1},
                    model_names={0: "item"},
                    task="detect",
                    kek=old_path.read_bytes(),
                    key_id=old_id,
                )
            original_destination = destination.read_bytes()
            rotate_kek(keys, kek_id="new")

            with self.assertRaises(FileExistsError):
                rotate_model_key(source, destination, key_dir=keys)

            self.assertEqual(original_destination, destination.read_bytes())


if __name__ == "__main__":
    unittest.main()
