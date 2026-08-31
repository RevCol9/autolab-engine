# 新环境部署指南

本文说明如何从 Git 部署 Django YOLO 训练服务，并与 Java 训练平台（boot-vision）联调。

## 架构概览

```
boot-vision.jar (:8801)          Django (:8008)              mamba yolo env
       │                                │                           │
       │  POST /api/v1/yolo_detector/   │  subprocess               │
       └──────────────────────────────►│ closed_loop_train.py ◄────┘
                                        │
storage/algorithms/{类型}/             │
  ├── models/baseline/weights/best.pt  ◄── Java「同步基线 PT」
  └── train{N}/weights/best.pt        ◄── 训练产出

/home/model/zhuqiang/{code}/weights/best.pt  ◄── 分析平台线上 PT（回灌目标）
```

## 1. 前置条件

| 项 | 要求 |
|----|------|
| OS | Linux（推荐，与 104 一致） |
| GPU | NVIDIA + 驱动（训练需 CUDA） |
| Python | 3.10+（104 为 3.12） |
| Git | 可访问 GitLab |
| 可选 | micromamba / mamba（闭环训练推荐） |

## 2. 克隆代码

```bash
git clone http://192.168.14.12:8040/niii-training-algorithm/niii-training-algorithm.git
cd niii-training-algorithm
```

## 3. 目录规划

建议与 104 保持一致（也可自定义，通过环境变量覆盖）：

| 路径 | 用途 |
|------|------|
| `/niii_machine_version/AI_trainning_platform/boot-vison-python-master` | 本项目（Django） |
| `/niii_machine_version/AI_trainning_platform/storage` | 训练样本、基线 PT、run 输出 |
| `/home/model/zhuqiang` | 分析平台算法仓库 |

创建 storage：

```bash
mkdir -p /niii_machine_version/AI_trainning_platform/storage/algorithms
```

## 4. 配置环境变量

复制并按实际路径修改：

```bash
cp deploy/paths.env.example /etc/niii/training-algorithm.env
# 编辑 NIII_TRAINING_STORAGE / NIII_ALGORITHM_ROOT / NIII_YOLO_PYTHON
```

启动前加载：

```bash
set -a && source /etc/niii/training-algorithm.env && set +a
```

代码通过 `niii_paths.py` 读取上述变量，默认值即 104 路径。

## 5. 安装 Django 服务环境

```bash
cd /niii_machine_version/AI_trainning_platform/boot-vison-python-master
python3 -m venv mydjango_project/venv
source mydjango_project/venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

**GPU 版 PyTorch**（按 CUDA 版本选择，104 为 cu124）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

验证 Django：

```bash
python manage.py check
```

## 6. 安装闭环训练环境（mamba 推荐）

104 上 `closed_loop_train.py` 由独立 mamba 环境执行（ultralytics 8.3.x）：

```bash
micromamba create -n yolo python=3.12 -y
micromamba activate yolo
pip install -r requirements-yolo.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

确认路径与 `NIII_YOLO_PYTHON` 一致，例如：

```bash
which python
# /root/.local/share/mamba/envs/yolo/bin/python
```

验证：

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

## 7. 预训练权重（Git 已含）

仓库内 `resources/yolov8n.pt` 为 Ultralytics 官方预训练（约 6MB），可选复制：

```bash
cp resources/yolov8n.pt algorithm_model/yolo/yolov8n.pt
cp resources/yolov8n.pt yolov8n.pt
```

**业务基线 best.pt 不在 Git 中**，见 `models/README.md`。

## 8. 准备业务基线 PT

任选一种方式：

### 方式 A：从分析平台算法仓库拷贝（新环境首次）

```bash
mkdir -p storage/algorithms/Helmet/models/baseline/weights
cp /home/model/zhuqiang/yolo_detect/weights/best.pt \
   storage/algorithms/Helmet/models/baseline/weights/best.pt
```

### 方式 B：通过训练平台 UI（推荐，与生产一致）

1. 部署 Java boot-vision.jar，配置 `algorithm-repository-path`
2. 打开「算法训练」→「同步基线 PT」
3. Java 会从算法仓库复制到 storage 基线路径

## 9. 启动训练 HTTP 服务（:21011，替代 Django :8008）

```bash
sudo mkdir -p /etc/niii
sudo cp /project/kang/autolab-engine/training/paths.env.example /etc/niii/training-algorithm.env
# 编辑 NIII_TRAINING_STORAGE / NIII_YOLO_PYTHON 后：
sudo cp /project/kang/autolab-engine/deploy/niii-autolab-train.service.example \
  /etc/systemd/system/niii-autolab-train.service
sudo systemctl daemon-reload
sudo systemctl enable --now niii-autolab-train.service
```

验证：

```bash
curl -s http://127.0.0.1:21011/api/health
curl -s -X POST http://127.0.0.1:21011/api/v1/yolo_detector/train/detection \
  -H 'Content-Type: application/json' -d '{"action":"stop"}'
```

旧 Django :8008 可逐步下线；训练 API 路径与字段保持兼容。

## 10. 配置 Java 训练平台

`boot-vision` 配置（`application-dev.yml`）：

```yaml
jeecg:
  path:
    upload: /niii_machine_version/AI_trainning_platform/storage

niii:
  closed-loop:
    train-service-url: http://127.0.0.1:21011
    algorithm-repository-path: /home/model/zhuqiang
```

## 11. 端到端验证

1. 训练平台 → 误报样本库有负样本  
2. 同步基线 PT 成功  
3. 启动误报优化训练 → Django 日志有 `closed_loop_train`  
4. `storage/algorithms/Helmet/trainN/trainning_data.csv` 出现并更新  
5. 训练成功后 Java 归档版本，手动回灌到算法仓库  

## 12. 开发机同步到 104

Windows：

```powershell
cd AI训练平台\niii_training_algorithm
git push
.\deploy\sync-to-104.ps1
```

Linux：

```bash
./deploy/sync-to-104.sh
```

同步**不会覆盖** 104 上的 venv、业务 best.pt、训练输出。

## 常见问题

| 现象 | 处理 |
|------|------|
| `请先同步基线 PT` | storage 基线路径无 best.pt，执行同步或手动拷贝 |
| `closed_loop_train failed` | 检查 `NIII_YOLO_PYTHON`、GPU、ultralytics 版本 |
| 路径错误 | 检查 `/etc/niii/training-algorithm.env` |
| SAM 功能报错 | SAM 权重需单独下载到 `algorithm_model/SAM/pretrained_checkpoint/` |

## 相关文档

- 权重分层说明：`models/README.md`
- bootstrap 权重：`resources/README.md`
- 路径示例：`deploy/paths.env.example`
