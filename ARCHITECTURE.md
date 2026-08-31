# autolab-engine 架构说明

详细说明见本地 [`docs/architecture.md`](docs/architecture.md)、[`docs/data-transfer.md`](docs/data-transfer.md)、[`docs/roadmap.md`](docs/roadmap.md)（`docs/` 不入库）。

本仓库提供两个独立 HTTP 服务，由 systemd 分别管理：

| 服务 | 入口 | 默认端口 | 配置文件 |
|------|------|----------|----------|
| 推理/标注 | `run.py` → `api/inference.py` | 21010 | `config/annotation.yaml` |
| 训练 | `training/run.py` → `api/train.py` | 21011 | `config/training/base.yaml` + `detection.yaml` |

环境变量：`ANNOTATION_CONFIG_PATH`（推理）、`TRAINING_TASK`（`detection` / `segmentation`）、`TRAINING_CONFIG_PATH`（覆盖任务 yaml）。

## 配置目录

```
config/
├── annotation.yaml              # 推理：models、gpu、locate、postprocess（不入库）
├── annotation.example.yaml
└── training/
    ├── base.yaml                # 训练共用：server、device、gpu
    ├── base.example.yaml
    ├── detection.yaml           # 检测训练 Ultralytics 默认超参
    ├── detection.example.yaml
    ├── segmentation.yaml        # 分割训练 Ultralytics 默认超参
    └── segmentation.example.yaml
```

## 目录职责

```
autolab-engine/
├── run.py
├── config/                     # 训推配置统一目录
├── annotation/
│   ├── settings.py
│   ├── registry.py
│   ├── predict_core.py
│   └── engines/
├── api/
│   ├── inference.py
│   └── train.py
├── training/
│   ├── settings.py
│   ├── service.py
│   └── closed_loop_train.py
└── shared/
    ├── config_yaml.py
    └── gpu_lock.py
```

## 推理流程

```
POST /api/predict (api/inference.py)
  → predict_core.run_predict_locked
      → threading.RLock（进程内串行）
      → registry.cross_gpu_session（fcntl 文件锁，按 GPU 索引）
      → registry.ensure_model_engine
      → engines/*.predict
      → box_format / mask_format 归一化
```

## 训练流程

```
POST /api/v1/.../train/detection action=start (api/train.py)
  → service.JobManager.start
      → GpuDeviceLock.acquire（非阻塞；失败则 409）
      → hparams.write_job_train_config → save_dir/train_config.yaml
      → trainer.popen_detection_train → closed_loop_train.py
```

进度：`GET /api/train/progress`；日志：`{save_dir}/train.log`。

分割训练：`POST .../train/segmentation`，后端插件 `training/backends/segmentation.py`。

## 配置分工

| 文件 | 内容 |
|------|------|
| `config/annotation.yaml` | `server`、各 `models[].device`、`gpu`、`locate`、`postprocess` |
| `config/training/base.yaml` | 训练 `server`、`device`、`gpu` |
| `config/training/detection.yaml` | `epochs`/`batch`/… Ultralytics 默认项 |

双卡训推分离：推理 `models[].device: cuda:0`，训练 `base.yaml` 里 `device: "1"`。

## GPU 锁（shared/gpu_lock.py）

- 锁文件：`{NIII_GPU_LOCK_DIR|/var/run}/niii-gpu{N}.lock`
- 推理：`cross_gpu_session` 在 load/predict 期间持锁
- 训练：`JobManager` 在整段训练期间持锁
