import numpy as np


def clip_box(box, W, H):
    x1 = int(np.clip(box[0], 0, W - 1))
    y1 = int(np.clip(box[1], 0, H - 1))
    x2 = int(np.clip(box[2], 0, W - 1))
    y2 = int(np.clip(box[3], 0, H - 1))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 == x1: x2 = min(x1 + 1, W - 1)
    if y2 == y1: y2 = min(y1 + 1, H - 1)
    return [x1, y1, x2, y2]


def clamp_delta(prev, new, max_delta, W, H):
    """각 변 이동량을 ±max_delta로 제한"""
    px1, py1, px2, py2 = prev
    nx1, ny1, nx2, ny2 = new
    nx1 = int(np.clip(nx1, px1 - max_delta, px1 + max_delta))
    ny1 = int(np.clip(ny1, py1 - max_delta, py1 + max_delta))
    nx2 = int(np.clip(nx2, px2 - max_delta, px2 + max_delta))
    ny2 = int(np.clip(ny2, py2 - max_delta, py2 + max_delta))
    return clip_box([nx1, ny1, nx2, ny2], W, H)


def apply_ops(prev_box, ops, W, H, max_delta_px):
    """ops를 prev_box에 적용하여 후보 박스 반환."""
    if not isinstance(ops, dict) or "type" not in ops:
        return prev_box
    x1, y1, x2, y2 = prev_box
    bw, bh = (x2 - x1), (y2 - y1)
    t = str(ops.get("type", "")).lower()

    if t == "delta":
        dx = int(ops.get("dx", 0))
        dy = int(ops.get("dy", 0))
        dl = int(ops.get("dl", 0))
        dr = int(ops.get("dr", 0))
        dt = int(ops.get("dt", 0))
        db = int(ops.get("db", 0))
        cand = [x1 + dx + dl, y1 + dy + dt, x2 + dx + dr, y2 + dy + db]
        cand = clip_box(cand, W, H)
        cand = clamp_delta(prev_box, cand, max_delta_px, W, H)
        return cand

    if t in ("expand", "shrink"):
        amount = int(ops.get("amount", 0))
        amount = abs(amount) if t == "expand" else -abs(amount)
        cand = [x1 - amount, y1 - amount, x2 + amount, y2 + amount]
        cand = clip_box(cand, W, H)
        cand = clamp_delta(prev_box, cand, max_delta_px, W, H)
        return cand

    if t == "recenter":
        target = ops.get("target", None)
        if isinstance(target, (list, tuple)) and len(target) == 2:
            tcx = int(target[0]); tcy = int(target[1])
            half_w = max(1, bw // 2); half_h = max(1, bh // 2)
            cand = [tcx - half_w, tcy - half_h, tcx + half_w, tcy + half_h]
            cand = clip_box(cand, W, H)
            cand = clamp_delta(prev_box, cand, max_delta_px, W, H)
            return cand

    return prev_box


def denorm_xywh_to_xyxy(norm_xywh, img_w, img_h):
    """정규화된 xywh -> 픽셀 xyxy 변환 (music_solo, music_duet, vggss_duet GT용)"""
    x, y, w, h = norm_xywh
    x1 = int(round(x * img_w))
    y1 = int(round(y * img_h))
    x2 = int(round((x + w) * img_w))
    y2 = int(round((y + h) * img_h))
    return [x1, y1, x2, y2]


def denorm_xyxy_to_xyxy(norm_xyxy, img_w, img_h):
    """정규화된 xyxy -> 픽셀 xyxy 변환 (vggss_single GT용)"""
    x1 = int(round(norm_xyxy[0] * img_w))
    y1 = int(round(norm_xyxy[1] * img_h))
    x2 = int(round(norm_xyxy[2] * img_w))
    y2 = int(round(norm_xyxy[3] * img_h))
    return [x1, y1, x2, y2]


def iou_xyxy(a, b):
    """두 박스의 IoU 계산 (duet 매칭용)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter + 1e-8
    return inter / union
