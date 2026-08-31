#!/usr/bin/env python3
"""Ultralytics 闭环训练子进程（由 trainer.popen_train 启动）。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from training.backends import get_backend
from training.hparams import load_job_config
from training.reporting import (
    build_report,
    evaluate_model,
    utc_now,
    utc_now_from_seconds,
    write_csv,
    write_json,
)
from training.resource_sampler import ResourceSampler, capture_resource_snapshot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to train_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_job_config(args.config)
    train_task = str(config.pop("train_task"))
    backend = get_backend(train_task)
    save_dir = Path(str(config.pop("save_dir")))
    model_path = str(config.pop("model"))
    data_path = str(config.pop("data"))
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "trainning_data.csv"
    process_start = time.time()
    train_start = process_start
    sampler = None
    state = {"epoch_start": train_start, "header_written": False}
    baseline_eval = {}
    trained_eval = {}
    actual_resource = {}
    sampler_started = False
    device = str(config.get("device", "0"))

    def on_fit_epoch_end(trainer):
        now = time.time()
        epoch = int(trainer.epoch) + 1
        epochs = int(trainer.epochs)
        epoch_spend = now - state["epoch_start"]
        state["epoch_start"] = now
        total_spend = now - train_start
        remaining = total_spend * (epochs - epoch) / epoch if epoch > 0 else 0
        metrics = trainer.metrics or {}
        resource = sampler.snapshot() if sampler else capture_resource_snapshot(device=device)
        row = backend.build_epoch_row(
            trainer,
            metrics,
            resource,
            epoch_index=epoch - 1,
            epochs=epochs,
            epoch_spend=epoch_spend,
            total_spend=total_spend,
            remaining=remaining,
        )
        write_csv(csv_path, row, state)

    try:
        model = YOLO(model_path)
        batch = int(config.get("batch", 4))
        imgsz = int(config.get("imgsz", 640))
        device = str(config.get("device", "0"))
        baseline_eval = evaluate_model(
            model_path=model_path,
            data=data_path,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project_dir=save_dir,
            run_name="baseline_eval",
        )
        train_start = time.time()
        state["epoch_start"] = train_start
        sampler = ResourceSampler(device=device).start()
        sampler_started = True
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(data=data_path, **config)
        trained_pt = save_dir / "weights" / "best.pt"
        actual_resource = sampler.stop()
        sampler_started = False
        if trained_pt.is_file():
            trained_eval = evaluate_model(
                model_path=str(trained_pt),
                data=data_path,
                batch=batch,
                imgsz=imgsz,
                device=device,
                project_dir=save_dir,
                run_name="trained_eval",
            )
        else:
            trained_eval = {"error": "trained weights not found", "modelPath": str(trained_pt)}
    finally:
        if sampler_started:
            actual_resource = sampler.stop()
            sampler_started = False
        finished_at = utc_now()
        report = build_report(
            config={
                "train_task": train_task,
                "model": model_path,
                "data": data_path,
                **config,
            },
            save_dir=save_dir,
            csv_path=csv_path,
            started_at=utc_now_from_seconds(train_start),
            finished_at=finished_at,
            actual_resource=actual_resource,
            baseline_eval=baseline_eval,
            trained_eval=trained_eval,
            run_duration_sec=max(0, int(round(time.time() - process_start))),
        )
        write_json(save_dir / "report.json", report)


if __name__ == "__main__":
    main()
