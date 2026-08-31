# autolab-engine 架构说明

本仓库提供两个独立 HTTP 服务，由 systemd 分别管理：

| 服务 | 入口 | 默认端口 | 配置文件 |
|------|------|----------|----------|
| 推理/标注 | `run.py` → `api/inference.py` | 21010 | `annotation/config.yaml` |
| 训练 | `training/run.py` → `api/train.py` | 21011 | `training/config.yaml` |

环境变量：`ANNOTATION_CONFIG_PATH` 或 `CONFIG_PATH`（推理）、`TRAINING_CONFIG_PATH`（训练）。

## 目录职责

```
autolab-engine/
├── run.py                      # 推理 uvicorn 入口
├── annotation/
│   ├── config.yaml             # 推理：models、gpu、locate、postprocess（不入库）
│   ├── settings.py             # YAML → Settings
│   ├── registry.py             # 引擎缓存 + GPU 锁会话
│   ├── predict_core.py         # 单图/批量推理逻辑
│   └── engines/                # YOLO、Locate、SAM3
├── api/
│   ├── inference.py            # 推理 HTTP
│   └── train.py                # 训练 HTTP
├── training/
│   ├── config.yaml             # 训练：device、超参、server（不入库）
│   ├── service.py              # JobManager
│   ├── hparams.py              # API JSON → train_config.yaml
│   ├── trainer.py              # Popen 子进程
│   └── closed_loop_train.py    # Ultralytics 训练循环
└── shared/gpu_lock.py          # 跨进程 GPU 文件锁
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

Locate 引擎在 `box_postprocess.refine_detections` 中做非法框过滤、NMS、狂出框检测。

## 训练流程

```
POST /api/v1/.../train/detection action=start (api/train.py)
  → service.JobManager.start
      → GpuDeviceLock.acquire（非阻塞；失败则 409）
      → hparams.write_job_train_config → save_dir/train_config.yaml
      → trainer.popen_detection_train → closed_loop_train.py
      → 后台线程 wait → collect_train_result（校验 best.pt）
```

进度：`GET /api/train/progress` 读 `trainning_data.csv`；日志在 `{save_dir}/train.log`。

## 配置分工

| 文件 | 内容 |
|------|------|
| `annotation/config.yaml` | `server`、各 `models[].device`、`gpu`、`locate`、`postprocess` |
| `training/config.yaml` | `server`、`device`、`gpu`、`epochs`/`batch`/… Ultralytics 默认项 |

双卡训推分离示例：推理 `models[].device: cuda:0`，训练 `device: "1"`。仅**同一 GPU 索引**上训推通过文件锁互斥；分卡可并行。

## GPU 锁（shared/gpu_lock.py）

- 锁文件：`{NIII_GPU_LOCK_DIR|/var/run}/niii-gpu{N}.lock`
- 推理：`cross_gpu_session` 在 load/predict 期间持锁（可配置超时 → HTTP 503）
- 训练：`JobManager` 在整段训练期间持锁
- 进程崩溃时 fd 关闭，锁自动释放
