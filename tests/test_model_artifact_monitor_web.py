from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

import training.artifact_monitor_service as monitor_module
from api import artifact_monitor as monitor_api
from training.artifact_monitor_service import (
    ArtifactMonitorBusyError,
    ArtifactMonitorService,
)


class _FakeMonitorProcess:
    def __init__(self, command: list[str]):
        self.command = command
        self.pid = 4321
        self.returncode = None
        self.ready_file = self._argument("--ready-file")
        self.stop_file = self._argument("--stop-file")
        self.event_log = self._argument("--events")
        self.summary = self._argument("--summary")
        self.ready_file.write_text('{"schemaVersion": 1}\n', encoding="utf-8")
        records = [
            {"sequence": 1, "kind": "monitor_started"},
            {"sequence": 2, "kind": "filesystem_event", "path": "best.niii-model"},
        ]
        self.event_log.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def _argument(self, name: str) -> Path:
        return Path(self.command[self.command.index(name) + 1])

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if not self.stop_file.exists():
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.summary.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "status": "passed",
                    "forbiddenEventCount": 0,
                }
            ),
            encoding="utf-8",
        )
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class ArtifactMonitorWebServiceTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.storage = self.base / "storage"
        self.run_root = self.storage / "project" / "task" / "train1"
        self.run_root.mkdir(parents=True)
        self.script = self.base / "monitor.py"
        self.script.write_text("# test monitor\n", encoding="utf-8")
        self.service = ArtifactMonitorService(
            storage_root=self.storage,
            evidence_root=self.base / "evidence",
            temp_root=self.base / "monitor-tmp",
            python_executable="python",
            monitor_script=self.script,
            ready_timeout_sec=0.1,
            stop_timeout_sec=0.1,
        )
        self.processes: list[_FakeMonitorProcess] = []

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _popen(self, command, **kwargs):
        kwargs["stdout"].write("monitor boot\n")
        kwargs["stdout"].flush()
        process = _FakeMonitorProcess(command)
        self.processes.append(process)
        return process

    def test_start_events_environment_and_stop(self):
        with patch.object(monitor_module.sys, "platform", "linux"), patch.object(
            monitor_module.subprocess,
            "Popen",
            side_effect=self._popen,
        ):
            started = self.service.start(
                project_id="project",
                task_id="task",
                train_num="train1",
            )
            self.assertEqual("running", started["status"])
            self.assertEqual(str(self.run_root.resolve()), started["roots"][0])

            environment = self.service.environment_for(self.run_root)
            self.assertEqual(started["tempDir"], environment["TMPDIR"])
            self.assertEqual(environment["TMPDIR"], environment["TMP"])
            self.assertEqual(environment, started["environment"])
            self.assertTrue(Path(environment["TMPDIR"]).is_dir())
            self.assertTrue(Path(environment["TORCH_HOME"]).is_dir())

            page = self.service.events(after_sequence=1, limit=10)
            self.assertEqual(1, page["count"])
            self.assertEqual(2, page["events"][0]["sequence"])
            process_log = self.service.process_log(tail=10)
            self.assertEqual(["monitor boot"], process_log["lines"])

            stopped = self.service.stop()
            self.assertEqual("passed", stopped["status"])
            self.assertEqual(0, stopped["monitorExitCode"])
            self.assertTrue(self.processes[0].stop_file.is_file())

    def test_second_active_session_is_rejected(self):
        with patch.object(monitor_module.sys, "platform", "linux"), patch.object(
            monitor_module.subprocess,
            "Popen",
            side_effect=self._popen,
        ):
            self.service.start(
                project_id="project",
                task_id="task",
                train_num="train1",
            )
            with self.assertRaises(ArtifactMonitorBusyError):
                self.service.start(
                    project_id="project",
                    task_id="task",
                    train_num="train1",
                )

    def test_web_request_cannot_select_an_arbitrary_path(self):
        fields = set(monitor_api.ArtifactMonitorStartBody.model_fields)
        self.assertEqual({"projectId", "taskId", "trainNum"}, fields)
        with self.assertRaises(ValidationError):
            monitor_api.ArtifactMonitorStartBody(
                projectId="project",
                taskId="task",
                trainNum="train1",
                roots=["/etc"],
            )

    def test_busy_service_maps_to_http_conflict(self):
        body = monitor_api.ArtifactMonitorStartBody(
            projectId="project",
            taskId="task",
            trainNum="train1",
        )
        with patch.object(
            monitor_api.MONITOR_SERVICE,
            "start",
            side_effect=ArtifactMonitorBusyError("busy"),
        ):
            with self.assertRaises(HTTPException) as captured:
                monitor_api.start_artifact_monitor(body)
        self.assertEqual(409, captured.exception.status_code)

    def test_training_api_registers_monitor_routes(self):
        from api.train import app

        paths = set(app.openapi()["paths"])
        self.assertIn("/api/model-artifact-monitor/start", paths)
        self.assertIn("/api/model-artifact-monitor/status", paths)
        self.assertIn("/api/model-artifact-monitor/events", paths)
        self.assertIn("/api/model-artifact-monitor/process-log", paths)
        self.assertIn("/api/model-artifact-monitor/stop", paths)

        response = TestClient(app).get("/monitor/model-artifacts")
        self.assertEqual(200, response.status_code)
        self.assertIn("模型文件事件监控", response.text)
        self.assertIn("实时文件事件", response.text)
        self.assertIn("监控进程日志", response.text)
        self.assertEqual("no-store", response.headers["cache-control"])
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])


class TrainingMonitorEnvironmentTest(unittest.TestCase):
    def test_training_subprocess_receives_monitored_temp_directory(self):
        from training import trainer

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            process = Mock(pid=1234)
            with patch.object(
                trainer,
                "build_closed_loop_cmd",
                return_value=["python", "train.py"],
            ), patch.object(
                trainer,
                "train_save_dir",
                return_value=save_dir,
            ), patch.object(
                trainer,
                "reset_run_artifacts",
            ), patch.object(
                trainer.MONITOR_SERVICE,
                "environment_for",
                return_value={"TMPDIR": "/watched/tmp"},
            ), patch.object(
                trainer.subprocess,
                "Popen",
                return_value=process,
            ) as popen:
                trainer.popen_train(
                    {"projectId": "p", "taskId": "t", "trainNum": "n"},
                    task="detection",
                )

            environment = popen.call_args.kwargs["env"]
            self.assertEqual("/watched/tmp", environment["TMPDIR"])
            self.assertEqual("1", environment["PYTHONUNBUFFERED"])


if __name__ == "__main__":
    unittest.main()
