#!/usr/bin/env python3
"""Run the independent Linux inotify model-artifact evidence process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.model_crypto.artifact_monitor import (  # noqa: E402
    ArtifactMonitorConfig,
    DEFAULT_FORBIDDEN_SUFFIXES,
    DEFAULT_MAX_EVENT_LOG_BYTES,
    LinuxInotifyArtifactMonitor,
    install_signal_handlers,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立 Linux inotify 模型明文工件监控"
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        required=True,
        help="待监控目录；可重复指定，启动前必须存在",
    )
    parser.add_argument("--events", type=Path, required=True, help="JSONL 事件证据")
    parser.add_argument("--summary", type=Path, required=True, help="最终 JSON 汇总")
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument(
        "--forbid-suffix",
        action="append",
        default=None,
        help="禁止文件后缀；可重复指定，默认使用模型明文后缀集合",
    )
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument(
        "--max-event-log-bytes",
        type=int,
        default=DEFAULT_MAX_EVENT_LOG_BYTES,
        help="事件日志容量上限；超限时 fail closed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = ArtifactMonitorConfig.build(
            roots=args.root,
            event_log=args.events,
            summary=args.summary,
            ready_file=args.ready_file,
            stop_file=args.stop_file,
            lock_file=args.lock_file,
            parent_pid=args.parent_pid,
            forbidden_suffixes=args.forbid_suffix or DEFAULT_FORBIDDEN_SUFFIXES,
            poll_interval_sec=args.poll_interval,
            max_event_log_bytes=args.max_event_log_bytes,
        )
        monitor = LinuxInotifyArtifactMonitor(config)
        install_signal_handlers(monitor)
        result = monitor.run()
    except Exception as exc:
        print(f"模型工件监控启动失败: {exc}", file=sys.stderr)
        return 1

    if result.passed:
        print(f"模型工件监控通过，汇总: {result.summary}")
        return 0
    print(
        "模型工件监控失败: "
        f"forbidden={result.forbidden_event_count}, "
        f"integrity={result.integrity_failure_count}；汇总: {result.summary}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
