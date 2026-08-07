# niii-training-algorithm

公司训练平台 **Django YOLO 训练服务**（原 `boot-vison-python-master`），现作为 **`autolab-engine` 仓库子目录** 维护，不再单独克隆 GitLab 仓库。

- **104 运行目录**: `/niii_machine_version/AI_trainning_platform/boot-vison-python-master`
- **半自动标注对接**: 见本目录 `deploy/annotation-engines.md`（与 `:21010` autolab-engine 协作）

## 目录说明

| 目录 | 作用 |
|------|------|
| `ml/` | Django API，`POST /api/v1/yolo_detector/train/detection` |
| `algorithm_model/yolo/` | YOLO 训练（`yolo_detect.py`、`closed_loop_train.py`） |
| `ML_backend/` | Django 配置与 wsgi 算法注册 |
| `resources/` | 可分发的小型 bootstrap 资源（含 `yolov8n.pt`） |
| `models/` | 权重分层说明文档 |
| `deploy/` | 部署与同步脚本 |

## 新环境部署

完整步骤见 **[deploy/setup-new-env.md](deploy/setup-new-env.md)**，包括：

- Django venv（`requirements.txt`）
- 闭环训练 mamba 环境（`requirements-yolo.txt`）
- storage / 算法仓库 / 基线 PT 准备
- 环境变量（`deploy/paths.env.example`）

## 在 autolab-engine 仓库中使用

```bash
# 克隆统一算法仓库（含推理 autolab-engine + 本训练服务）
git clone git@github.com:RevCol9/autolab-engine.git
cd autolab-engine/niii_training_algorithm
```

## 同步到 104

Windows：

```powershell
cd autolab-engine/niii_training_algorithm
.\deploy\sync-to-104.ps1
```

Linux：

```bash
./deploy/sync-to-104.sh
```

同步后 Django 自动重启（8008）。**不会覆盖** 104 上的 venv、业务 `best.pt`、训练输出。

## Git 中包含 / 不包含

| 纳入 Git | 不纳入 Git（见 `models/README.md`） |
|----------|-----------------------------------|
| Python 源码 | `mydjango_project/venv/` |
| `resources/yolov8n.pt`（bootstrap） | 业务基线 / 线上 `best.pt` |
| 部署文档与 requirements | 训练 run 输出、`trainning_data.csv` |
| | SAM 预训练权重、Miniconda 安装包 |

本目录沿用自己的 `.gitignore`；权重与大文件规则与仓库根目录 `weights/**` 一致，勿提交 secrets。

## 常见修改入口

- 闭环训练：`algorithm_model/yolo/closed_loop_train.py`
- 训练 API：`ml/views.py` → `TrainDetectionView`
- YOLO 封装：`algorithm_model/yolo/yolo_detect.py`
- 部署路径：`niii_paths.py` / 环境变量
