# autolab-engine

训练与自动标注服务。训练和推理分别由独立的 HTTP 进程运行；可配置的 YOLO 检测、
分割模型使用 `.niii-model` 加密权重。SAM3、LocateAnything 是独立的自动标注模块，
不在该加密范围内。

## 运行入口

| 服务 | 命令 | 默认端口 | 配置模板 |
|------|------|----------|----------|
| 推理/标注 | `python run.py` | 21010 | `config/annotation.example.yaml` |
| 训练 | `python training/run.py` | 21011 | `config/training/*.example.yaml` |

Python 依赖见 `requirements.txt`；启用 SAM3、LocateAnything 时还需按
`requirements-vlm.txt` 准备环境。Ultralytics 版本固定为 8.4.113。
复制配置模板为对应的 `.yaml` 后按部署环境修改；本地配置、密钥、模型文件和运行结果不入库。
服务部署示例在 `deploy/`。模块与调用链见 [ARCHITECTURE.md](ARCHITECTURE.md)，
模型存储约束见 [training/WEIGHTS.md](training/WEIGHTS.md)。

## 模型与密钥

生产训练和可配置 YOLO 推理只加载 `.niii-model`。初始可信 `.pt` 仅在隔离导入环境中
使用 `scripts/convert_trusted_model.py` 转换，转换脚本不进入生产发布包。
KEK 存放于仓库外的受限目录，通过 `NIII_MODEL_KEYS_DIR` 或独立配置定位；
不要将密钥或模型文件提交到 Git。

## 生成生产发布包

在已提交且测试通过的版本上执行：

```sh
python scripts/build_release.py
```

脚本只归档当前 `HEAD` 的运行时文件，生成 `dist/autolab-engine-<commit>.tar.gz`，
并扫描归档条目；发现 `.git`、明文模型、密钥文件或离线导入脚本时会失败。
部署时解压该包，单独注入配置、KEK 和 `.niii-model`，不要把 Git 工作目录或 `.git`
复制到生产服务器。发布包扫描只覆盖包本身；上线前还需检查模型卷、工作目录和缓存，
并用文件事件监控验证训练全过程。

当前 Git 历史中曾出现明文 `.pt`；生产发布必须使用上述不含 `.git` 的发布包。
