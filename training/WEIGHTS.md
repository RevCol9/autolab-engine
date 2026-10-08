# 模型权重说明

生产训练与 YOLO 推理只允许 `.niii-model`。仓库、部署镜像、模型卷和训练目录不得包含
业务 `.pt`/`.pth`，也不得采用“先写 `best.pt`，再加密删除”的流程。

## 训练 storage

路径根：`{STORAGE_ROOT}/algorithms/{算法类型}/`

| 路径 | 用途 |
|------|------|
| `models/baseline/weights/best.niii-model` | 误报微调的加密起始权重 |
| `train{N}/weights/best.niii-model` | 第 N 次训练的加密最佳权重 |
| `train{N}/weights/last.niii-model` | 第 N 次训练的加密末轮权重 |
| `models/versions/v{N}/best.niii-model` | Java 后端归档/回灌版本 |

`is_continue` 加载上次训练的 `weights/best.niii-model` 并创建新 optimizer，不等同于
精确 checkpoint resume；首版明确拒绝 `resume=True`。

## 模型来源

- 从零训练使用受控的 YOLOv8、YOLO11、YOLO26 检测/分割架构 YAML，不自动下载权重。
- 预训练和继续微调必须由后端提供允许根目录内的 `.niii-model`。
- 初始 `.pt` 只允许在部署环境外的隔离导入机进行可信转换；转换后只发布密文容器。
- KEK 与模型分离，通过受限密钥目录和 key ID 注入，不提交 Git、不放入模型目录。

上线前还必须完成 GPU/业务权重验证、全过程文件创建事件监控、异常故障注入和发布物
扫描。SAM3、LocateAnything 等尚未接入 `.niii-model` 的入口不得包含在“服务器无明文
模型”的生产部署中。
