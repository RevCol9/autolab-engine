# resources/

本目录存放**可随 Git 分发**的小型 bootstrap 资源。

## yolov8n.pt

- Ultralytics YOLOv8n 官方 COCO 预训练权重（约 6.3 MB）
- 用途：Legacy 从零训练、离线环境兜底、开发测试
- **误报闭环微调**仍依赖 storage 中的业务基线 `best.pt`（见 `models/README.md`）

部署后可选：复制到项目根或 `algorithm_model/yolo/`，供旧代码路径引用：

```bash
cp resources/yolov8n.pt algorithm_model/yolo/yolov8n.pt
cp resources/yolov8n.pt yolov8n.pt
```

Ultralytics 也可在首次使用时自动下载同名文件；纳入 Git 是为了无公网或内网部署场景。
