# ⚠️ 已废弃（仅作历史对照）

本目录为**原公司 Django 训练服务**快照，**不再参与开发与部署**。

请改用：

| 用途 | 位置 |
|------|------|
| 训练 HTTP（FastAPI `:21011`） | [`training/`](../training/) |
| 推理 HTTP（FastAPI `:21010`） | [`app/`](../app/) + [`run.py`](../run.py) |
| 训练接口文档 | [`docs/10-training-api.md`](../docs/10-training-api.md) |
| 训练架构 | [`docs/09-training-module.md`](../docs/09-training-module.md) |
| systemd 部署 | [`deploy/niii-autolab-train.service.example`](../deploy/niii-autolab-train.service.example) |
| 半自动标注对接 | [`deploy/annotation-engines.md`](../deploy/annotation-engines.md) |

**不要**再启动本目录下的 Django `:8008`；训练由 `niii-autolab-train.service` 提供。
