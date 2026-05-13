import json


def load_music_solo_gt(gt_json_path):
    """
    music_solo.json 포맷: [{"file": "<vid>", "bbox": [x, y, w, h]}, ...]
    bbox: 0~1 정규화, xywh
    반환: vid -> norm_xywh
    """
    with open(gt_json_path, "r") as f:
        data = json.load(f)
    gt_dict = {}
    for item in data:
        vid = item["file"]
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            gt_dict[vid] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    return gt_dict


def load_vggss_gt(gt_json_path):
    """
    vggss.json 포맷: [{"file": "<vid>", "bbox": [[x1,y1,x2,y2], ...]}, ...]
    bbox: 0~1 정규화, xyxy, 첫 박스 사용
    반환: vid -> norm_xyxy
    """
    with open(gt_json_path, "r") as f:
        data = json.load(f)
    gt_dict = {}
    for item in data:
        vid = item["file"]
        bboxes = item.get("bbox", [])
        if isinstance(bboxes, list) and len(bboxes) > 0 and len(bboxes[0]) == 4:
            gt_dict[vid] = [float(v) for v in bboxes[0]]
    return gt_dict


def load_music_duet_gt(gt_json_path):
    """
    music_duet.json 포맷:
    [{"file": "<vid>", "bbox_src1": [x,y,w,h], "bbox_src2": [x,y,w,h],
      "class_src1": "...", "class_src2": "..."}, ...]
    반환: vid -> {"classes": [...], "boxes_xywh": [...]}
    """
    with open(gt_json_path, "r") as f:
        data = json.load(f)
    gt = {}
    for it in data:
        vid = it["file"]
        b1 = it.get("bbox_src1")
        b2 = it.get("bbox_src2")
        cls1 = it.get("class_src1", "unknown")
        cls2 = it.get("class_src2", "unknown")
        bx1 = b1 if (isinstance(b1, (list, tuple)) and len(b1) == 4) else None
        bx2 = b2 if (isinstance(b2, (list, tuple)) and len(b2) == 4) else None
        gt[vid] = {"classes": [cls1, cls2], "boxes_xywh": [bx1, bx2]}
    return gt


def load_vggss_duet_gt(gt_json_path):
    """
    vggss_duet.json 포맷:
    [{"file": "<vid>", "bbox_src1": [x,y,w,h], "bbox_src2": [x,y,w,h],
      "class_src1": "...", "class_src2": "..."}, ...]
    반환: vid -> {"classes": [...], "boxes_xywh": [...]}
    """
    with open(gt_json_path, "r") as f:
        data = json.load(f)
    gt = {}
    for it in data:
        vid = it["file"]
        b1 = it.get("bbox_src1")
        b2 = it.get("bbox_src2")
        cls1 = it.get("class_src1", "unknown")
        cls2 = it.get("class_src2", "unknown")
        bx1 = b1 if (isinstance(b1, (list, tuple)) and len(b1) == 4) else None
        bx2 = b2 if (isinstance(b2, (list, tuple)) and len(b2) == 4) else None
        gt[vid] = {"classes": [cls1, cls2], "boxes_xywh": [bx1, bx2]}
    return gt
