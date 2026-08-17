# 模型权重说明

本仓库**不包含**业务基线 `best.pt`。权重按用途分三层管理：

## 1. Git 仓库内（`resources/`）

| 文件 | 用途 |
|------|------|
| `resources/yolov8n.pt` | Ultralytics 官方预训练权重（约 6MB），用于从零训练或离线环境兜底 |

闭环误报微调**不直接使用**此文件，而是从 storage 基线 `best.pt` 继续训练。

## 2. 训练 storage（不进 Git）

路径根：`{STORAGE_ROOT}/algorithms/{算法类型}/`

| 路径 | 用途 | 如何获得 |
|------|------|----------|
| `models/baseline/weights/best.pt` | **误报微调起始权重** | 训练平台「同步基线 PT」 |
| `train{N}/weights/best.pt` | 第 N 次训练产出 | 训练成功后自动生成 |
| `models/versions/v{N}/best.pt` | 归档版本 | Java 后端归档 |

104 默认 storage 根：`/niii_machine_version/AI_trainning_platform/storage`

## 3. 分析平台算法仓库（不进 Git）

| 路径 | 用途 |
|------|------|
| `{algorithm-repository}/{algorithm-code}/weights/best.pt` | **线上推理 PT** |
| `.../weights/baseline/best.pt` | 基线快照（同步时写入） |
| `.../weights/backups/` | 回灌前自动备份 |
| `.../weights/versions/v{N}/best.pt` | 回灌用归档 |

104 默认算法仓库：`/home/model/zhuqiang`（如 `yolo_detect/weights/best.pt`）

## 新环境最小要求

1. 克隆本 Git 仓库并安装 Python 环境（见 `deploy/setup-new-env.md`）
2. 创建 storage 目录
3. 确保算法仓库存在线上 `best.pt`，或在 storage 中手动放置基线 PT
4. 在训练平台点击「同步基线 PT」，将线上模型复制到 storage 基线路径

## 不要提交进 Git 的文件

- 业务 `best.pt`（会随回灌/训练频繁变化）
- 训练 run 输出 `train*/weights/`
- SAM 预训练 `.pth`（体积大，按需下载）
- `yolov8m/s/seg.pt` 等 Legacy 权重
