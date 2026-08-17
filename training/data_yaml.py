"""classes.txt → YOLO data yaml（从原 ml/views.py txt_to_yaml 移植）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def classes_txt_to_yaml(
    *,
    dataset_root: str | Path,
    train_images_rel: str,
    classes_txt: str | Path,
    val_images_rel: str | None = None,
    yaml_path: str | Path | None = None,
) -> Path:
    """将 classes.txt 写成 Ultralytics 可用的 data yaml。

    Args:
        dataset_root: yaml 里的 path（绝对或相对均可，与原平台一致时多为 storage/...）
        train_images_rel: 相对 path 的 train 图像目录
        classes_txt: 每行一个类名
        val_images_rel: 默认与 train 相同（闭环误报场景常用）
        yaml_path: 输出路径，默认与 classes.txt 同目录、后缀 .yaml
    """
    txt_file = Path(classes_txt)
    out = Path(yaml_path) if yaml_path else txt_file.with_suffix(".yaml")
    names: Dict[int, str] = {}
    with txt_file.open("r", encoding="utf-8") as f:
        i = 0
        for line in f:
            line = line.strip()
            if line:
                names[i] = line
                i += 1

    data: Dict[str, Any] = {
        "path": str(dataset_root).strip(),
        "train": train_images_rel.strip(),
        "val": (val_images_rel or train_images_rel).strip(),
        "names": names,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return out


def prepare_data_yaml_for_job(param: Dict[str, Any]) -> Path:
    """按原 TrainThread 规则生成 classes.yaml。

    - projectId == algorithms：path=storage/algorithms/{taskId}，train={trainNum}/images
    - 其它：path=storage/{projectId}，train=images
    """
    from training.paths import STORAGE_ROOT, classes_txt_path, safe_id

    project_id = safe_id("projectId", param["projectId"])
    task_id = safe_id("taskId", param["taskId"])
    train_num = safe_id("trainNum", param.get("trainNum") or "train1")
    txt_path = classes_txt_path(project_id, task_id)
    if not txt_path.is_file():
        raise FileNotFoundError(f"classes.txt 不存在: {txt_path}")

    if project_id == "algorithms":
        yaml_base = STORAGE_ROOT / "algorithms" / task_id
        image_rel = f"{train_num}/images"
        return classes_txt_to_yaml(
            dataset_root=yaml_base,
            train_images_rel=image_rel,
            classes_txt=txt_path,
        )

    return classes_txt_to_yaml(
        dataset_root=STORAGE_ROOT / project_id,
        train_images_rel="images",
        classes_txt=txt_path,
    )
