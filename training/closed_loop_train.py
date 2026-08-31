#!/usr/bin/env python3
"""Closed-loop training via ultralytics 8.3+ (supports YOLO11 baseline PT).

由 training.trainer 通过 NIII_YOLO_PYTHON 子进程调用；产出 trainning_data.csv / report.json。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from training.hparams import load_job_config
from training.reporting import (
    build_report,
    evaluate_model,
    metric_lookup,
    trainer_lr,
    trainer_metric,
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
        row = {
            "epoch": epoch - 1,
            "train/box_loss": trainer_metric(trainer, "train/box_loss"),
            "train/cls_loss": trainer_metric(trainer, "train/cls_loss"),
            "train/dfl_loss": trainer_metric(trainer, "train/dfl_loss"),
            "metrics/precision(B)": metric_lookup(metrics, "precision"),
            "metrics/recall(B)": metric_lookup(metrics, "recall"),
            "metrics/mAP50(B)": metric_lookup(metrics, "map50"),
            "metrics/mAP50-95(B)": metric_lookup(metrics, "map5095"),
            "val/box_loss": metrics.get("val/box_loss", 0),
            "val/cls_loss": metrics.get("val/cls_loss", 0),
            "val/dfl_loss": metrics.get("val/dfl_loss", 0),
            "lr/pg0": trainer_lr(trainer, 0),
            "lr/pg1": trainer_lr(trainer, 1),
            "lr/pg2": trainer_lr(trainer, 2),
            "epoch_spend_time": round(epoch_spend, 4),
            "total_spend_time": round(total_spend, 4),
            "remaining_time": round(remaining, 4),
            "resource_cpu": resource.get("cpu", 0),
            "resource_mem": resource.get("mem", 0),
            "resource_gpu": resource.get("gpu", 0),
            "resource_gpu_mem_used_mb": resource.get("gpuMemUsedMb", 0),
            "resource_gpu_mem_total_mb": resource.get("gpuMemTotalMb", 0),
            "resource_disk": resource.get("disk", 0),
        }
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
