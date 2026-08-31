# autolab-engine 架构说明

本仓库提供两个独立 HTTP 服务，由 systemd 分别管理：

| 服务 | 入口 | 默认端口 | 配置文件 |
|------|------|----------|----------|
| 推理/标注 | `run.py` → `api/inference.py` | 21010 | **`annotation/config.yaml`** |
| 训练 | `training/run.py` → `api/train.py` | 21011 | **`training/config.yaml`** |

环境变量：`ANNOTATION_CONFIG_PATH` 或 `CONFIG_PATH`（推理）、`TRAINING_CONFIG_PATH`（训练）。

## 目录职责

```
autolab-engine/
├── run.py
├── annotation/
│   ├── config.yaml             # 推理：models、gpu、locate…（不入库）
│   ├── config.example.yaml
│   ├── settings.py             # 读取 annotation/config.yaml
│   └── engines/
├── training/
│   ├── config.yaml             # 训练：device、超参、server…（不入库）
│   ├── config.example.yaml
│   └── settings.py
├── api/                        # HTTP 路由
└── shared/gpu_lock.py
```

## 配置分工

| 文件 | 内容 |
|------|------|
| `annotation/config.yaml` | `server`（:21010）、`models[].device`、推理 `gpu`、`locate`、`postprocess` |
| `training/config.yaml` | `server`（:21011）、`device`、`gpu`、`epochs`/`batch`/… |

双卡训推分离：`annotation` 里 `models[].device: cuda:0`，`training/config.yaml` 里 `device: "1"`。

## GPU 互斥

`shared/gpu_lock.py` 按 GPU 索引加锁；仅同一索引时训推互斥，分卡可并行。
