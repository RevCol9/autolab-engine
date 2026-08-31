"""检测结果后处理：抑制把 max_new_tokens 额度打满的狂出框。

说明：
- max_new_tokens 只是生成上限，不是目标长度。
- 模型若未正常结束（未及时输出结束符），会持续吐出 <box>。
- 本模块在解析后做：非法框过滤 + 按类别 NMS，使接口返回更接近真实目标。
- 中文查询原文保留（用于标签）；英文任务模板中补充“勿重复、完成后停止”等约束。
"""

from __future__ import annotations

from typing import Any, Dict, List


def enhance_user_query(text: str, bilingual: bool = True) -> str:
    """整理用户查询文本。

    中文原样保留，不做自动翻译，避免改客户语义。
    bilingual 预留扩展（例如后续接词典/翻译），当前仅做 strip。
    """
    return (text or "").strip()


def build_detect_prompt(categories: List[str], bilingual: bool = True) -> str:
    """构造开放词汇检测提示词（类别列表）。

    注意：description 槽位只能放类别本身，不要附加英文说明，
    否则模型会把整段英文写进 <ref> 标签，前端标签条会铺满画面。
    """
    cats = "</c>".join(enhance_user_query(c, bilingual) for c in categories if c.strip())
    return f"Locate all the instances that matches the following description: {cats}."


def build_ground_multi_prompt(phrase: str, bilingual: bool = True) -> str:
    """构造短语多实例定位提示词。description 仅放用户原短语（可中文）。"""
    phrase = enhance_user_query(phrase, bilingual)
    return f"Locate all the instances that match the following description: {phrase}."


def build_ground_single_prompt(phrase: str, bilingual: bool = True) -> str:
    """构造短语单实例定位提示词。description 仅放用户原短语（可中文）。"""
    phrase = enhance_user_query(phrase, bilingual)
    return f"Locate a single instance that matches the following description: {phrase}."


def build_ground_text_prompt(phrase: str, bilingual: bool = True) -> str:
    """构造文本定位提示词。"""
    phrase = enhance_user_query(phrase, bilingual)
    return f"Please locate the text referred as {phrase}."


def build_ground_gui_prompt(phrase: str, output_type: str = "box", bilingual: bool = True) -> str:
    """构造 GUI 定位提示词；output_type=point 时改为指点。"""
    phrase = enhance_user_query(phrase, bilingual)
    if output_type == "point":
        return f"Point to: {phrase}."
    return f"Locate the region that matches the following description: {phrase}."


def build_point_prompt(phrase: str, bilingual: bool = True) -> str:
    """构造点定位提示词。"""
    phrase = enhance_user_query(phrase, bilingual)
    return f"Point to: {phrase}."


def _area(box: Dict[str, Any]) -> float:
    """计算框面积（像素）。"""
    return max(0.0, float(box["x2"]) - float(box["x1"])) * max(
        0.0, float(box["y2"]) - float(box["y1"])
    )


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """计算两个轴对齐框的 IoU。"""
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def filter_boxes(
    boxes: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    min_box_area_ratio: float = 0.0003,
    max_box_area_ratio: float = 0.85,
    min_side_px: float = 4.0,
) -> List[Dict[str, Any]]:
    """过滤非法几何、过小噪声框、几乎铺满全图的无效框，并裁剪到图像范围内。"""
    img_area = max(float(image_width * image_height), 1.0)
    kept: List[Dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
        if x2 <= x1 or y2 <= y1:
            continue
        # 裁剪到图像边界内
        x1 = min(max(x1, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        x2 = min(max(x2, 0.0), float(image_width))
        y2 = min(max(y2, 0.0), float(image_height))
        if x2 - x1 < min_side_px or y2 - y1 < min_side_px:
            continue
        area = (x2 - x1) * (y2 - y1)
        ratio = area / img_area
        if ratio < min_box_area_ratio or ratio > max_box_area_ratio:
            continue
        out = dict(box)
        out.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        kept.append(out)
    return kept


def nms_boxes(boxes: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """按 label 分组做非极大值抑制。

    同类高重叠框只保留面积较大者，用于压制狂出框时的近重复结果。
    最终按面积从大到小排序，便于前端优先展示主目标。
    """
    if not boxes:
        return []
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for box in boxes:
        by_label.setdefault(str(box.get("label") or ""), []).append(box)

    kept: List[Dict[str, Any]] = []
    for _label, group in by_label.items():
        group = sorted(group, key=_area, reverse=True)
        selected: List[Dict[str, Any]] = []
        for box in group:
            if all(_iou(box, s) < iou_threshold for s in selected):
                selected.append(box)
        kept.extend(selected)
    kept.sort(key=_area, reverse=True)
    return kept


def suspect_runaway_generation(raw_box_count: int, max_new_tokens: int, answer: str) -> bool:
    """判断是否像「打满 token 狂出框」。

    hybrid 解码大约每框约 6 个 token；若框数接近 max_new_tokens/6，
    基本可认定模型未正常结束，而是把额度用完了。
    """
    if max_new_tokens <= 0:
        return False
    approx_cap = max(8, int(max_new_tokens / 6))
    if raw_box_count >= int(approx_cap * 0.85):
        return True
    # 答案里堆积大量 box 串，也视为异常
    if raw_box_count >= 40 and answer.count("<box>") >= 40:
        return True
    return False


def clean_box_labels(
    boxes: List[Dict[str, Any]],
    preferred_label: str = "",
    max_label_chars: int = 32,
) -> List[Dict[str, Any]]:
    """清理异常长的 <ref> 标签。

    若模型把提示词英文后缀一并写进 ref，会导致前端标签绿条极宽、画面像被涂白。
    优先回退为用户原始短语 preferred_label。
    """
    preferred = (preferred_label or "").strip()
    out: List[Dict[str, Any]] = []
    for box in boxes:
        label = str(box.get("label") or "").strip()
        # 明显被污染：过长，或夹带英文约束句片段
        polluted = (
            len(label) > max_label_chars
            or "The query may be" in label
            or "Do not repeat" in label
            or "Stop when" in label
            or "Output each" in label
            or "Output exactly" in label
        )
        if polluted and preferred:
            label = preferred
        elif len(label) > max_label_chars:
            label = label[: max_label_chars - 1] + "…"
        item = dict(box)
        item["label"] = label
        out.append(item)
    return out


def refine_detections(
    boxes: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    answer: str = "",
    max_new_tokens: int = 256,
    enable: bool = True,
    nms_iou: float = 0.5,
    runaway_nms_iou: float = 0.35,
    min_box_area_ratio: float = 0.0003,
    max_box_area_ratio: float = 0.85,
    max_boxes: int = 64,
    preferred_label: str = "",
) -> Dict[str, Any]:
    """解析后的统一精炼入口。

    返回字段：
    - boxes: 精炼后的框
    - raw_box_count / kept_box_count: 精炼前后数量
    - runaway_suspected: 是否疑似打满 token
    - postprocess_applied: 是否执行了后处理
    - nms_iou_used: 实际使用的 IoU 阈值（狂跑时用更严的 runaway_nms_iou）
    - truncated_by_max_boxes: 是否因 max_boxes 安全上限被截断
    """
    raw_count = len(boxes)
    runaway = suspect_runaway_generation(raw_count, max_new_tokens, answer)

    if not enable:
        cleaned = clean_box_labels(boxes, preferred_label=preferred_label)
        return {
            "boxes": cleaned,
            "raw_box_count": raw_count,
            "kept_box_count": len(cleaned),
            "runaway_suspected": runaway,
            "postprocess_applied": False,
            "truncated_by_max_boxes": False,
        }

    filtered = filter_boxes(
        boxes,
        image_width,
        image_height,
        min_box_area_ratio=min_box_area_ratio,
        max_box_area_ratio=max_box_area_ratio,
    )
    filtered = clean_box_labels(filtered, preferred_label=preferred_label)
    # 疑似狂出框时用更严的 IoU，多合并近重复框
    iou = runaway_nms_iou if runaway else nms_iou
    suppressed = nms_boxes(filtered, iou_threshold=iou)

    truncated = False
    if max_boxes > 0 and len(suppressed) > max_boxes:
        suppressed = suppressed[:max_boxes]
        truncated = True

    return {
        "boxes": suppressed,
        "raw_box_count": raw_count,
        "kept_box_count": len(suppressed),
        "runaway_suspected": runaway,
        "postprocess_applied": True,
        "nms_iou_used": iou,
        "truncated_by_max_boxes": truncated,
    }
