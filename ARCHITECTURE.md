# autolab-engine 架构说明

本仓库提供两个独立 HTTP 服务，由 systemd 分别管理：

| 服务 | 入口 | 默认端口 | 职责 |
|------|------|----------|------|
| 推理 | `run.py` → `api/inference.py` | 21010 | YOLO / Locate / SAM3 在线推理 |
| 训练 | `training/run.py` → `api/train.py` | 21111（env 覆盖） | YOLO 闭环训练子进程调度 |

平台后端（boot-vision）通过 HTTP 调用上述两个端口，不直接 import 本仓库 Python 模块。

## 目录职责

```
autolab-engine/
├── run.py                      # 推理启动
├── config.yaml                 # 推理模型清单（本地，不入库）
├── api/                        # HTTP 层（仅 FastAPI 路由）
│   ├── inference.py            # 推理 REST API (:21010)
│   └── train.py                # 训练 REST API (:21111)
├── annotation/                 # 半自动标注 / 推理领域逻辑
│   ├── bootstrap.py            # 加载 config.yaml、应用环境变量
│   ├── registry.py             # 引擎缓存、load/unload
│   ├── predict_core.py         # 单图/批量推理逻辑
│   ├── runtime_state.py        # health 探测、unload 辅助
│   ├── settings.py             # config.yaml 解析
│   └── engines/                # YOLO、Locate、SAM3 引擎实现
├── training/                   # 训练领域逻辑（无 HTTP）
│   ├── run.py                  # 训练 HTTP 启动（挂载 api.train）
│   ├── service.py              # JobManager（单任务槽）
│   ├── trainer.py              # 子进程 popen / 杀进程
│   ├── hparams.py              # API → train_config.yaml
│   ├── closed_loop_train.py    # 子进程入口
│   ├── reporting.py            # CSV / report.json / 评估
│   ├── resource_sampler.py     # 训练期 GPU/CPU 采样
│   └── progress.py             # 解析 trainning_data.csv
├── shared/                     # 训推共用
│   └── gpu_lock.py             # 跨进程 GPU 文件锁
└── deploy/                     # systemd 模板
```

## 请求链路

### 推理

```
POST /api/predict
  → api/inference.py
  → annotation.predict_core.run_predict_locked
  → annotation.registry.ensure_model_engine
  → shared.gpu_lock（与训练互斥）
  → annotation.engines/*.predict
```

### 训练

```
POST /api/v1/yolo_detector/train/detection  (action=start)
  → api/train.py
  → training.service.JobManager.start
  → training.trainer.popen_detection_train
  → 子进程: closed_loop_train.py --config .../train_config.yaml
  → 产出: weights/best.pt, trainning_data.csv, report.json, train.log
```

## GPU 互斥

`shared/gpu_lock.py` 在推理 load/predict 与训练 `JobManager.start` 时各持一把同 index 的文件锁（`/var/run/niii-gpu{N}.lock`）。同一 GPU 上训推不能并行。

## 调试入口

| 现象 | 先看 |
|------|------|
| 推理 503 GPU 忙 | 训练是否在跑；`GET /api/health` 的 `gpu_locks` |
| 模型加载失败 | `config.yaml` 的 `models[].path`；权重文件是否存在 |
| 训练失败 | `{save_dir}/train.log`；`GET /api/train/progress` |
| 训练起不来 409 | 已有任务占用单槽位 JobManager |

## 相关文档

- 推理 API：`docs/02-api.md`、`docs/06-locate-sam3.md`、`docs/08-segment-api.md`
- 训练 API：`docs/10-training-api.md`、`docs/09-training-module.md`
- 部署：`deploy/README.md`
