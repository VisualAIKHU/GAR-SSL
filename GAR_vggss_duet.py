import os
import re
import json
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn import metrics

from model_utils import (
    load_qwen_omni_thinker, run_chat,
    parse_duet_locate_json, parse_duet_audio_json,
    parse_json_object, run_anchor_voting_multi,
)
from bbox_utils import clip_box, clamp_delta, apply_ops, denorm_xywh_to_xyxy, iou_xyxy
from evaluator import Evaluator
from data_utils import load_vggss_duet_gt
from prompts_duet import (
    build_prompt_locate_duet,
    build_prompt_audio_duet,
    build_prompt_analysis_anchor_voting_no_conf,
    build_prompt_refine,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAR - VGGSound Duet")
    parser.add_argument("--model_id",     type=str,   default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--frame_dir",    type=str,   default="/data/subin/VGGSound_duet/test/frames")
    parser.add_argument("--audio_dir",    type=str,   default="/data/subin/VGGSound_duet/test/audio")
    parser.add_argument("--gt_path",      type=str,   default="/data/subin/VGGSound_duet/vggss_duet_test.json")
    parser.add_argument("--out_root",     type=str,   default="/data/subin/GAR_CVPR26/outputs/GAR_vggss_duet")
    parser.add_argument("--cuda_device",  type=str,   default="0")
    parser.add_argument("--n_votes",      type=int,   default=7)
    parser.add_argument("--tau_av",       type=float, default=0.75)
    parser.add_argument("--tau_audio",    type=float, default=0.75)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    model, processor = load_qwen_omni_thinker(args.model_id)

    prompt_locate_A = build_prompt_locate_duet(dataset_name="VGGSS DUET")
    prompt_audio_B  = build_prompt_audio_duet(dataset_name="VGGSS DUET")

    frame_dir = args.frame_dir
    audio_dir = args.audio_dir
    gt_path   = args.gt_path

    out_root      = args.out_root
    vis_dir       = os.path.join(out_root, "vis_original")
    vis_dir_224   = os.path.join(out_root, "vis224")
    bbox_json_dir = os.path.join(out_root, "bbox")
    summary_json_path = os.path.join(out_root, "bbox_all.json")

    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(vis_dir_224, exist_ok=True)
    os.makedirs(bbox_json_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)

    gt_dict     = load_vggss_duet_gt(gt_path)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    print(f"Found {len(frame_files)} frames")

    evaluator = Evaluator()
    all_ap, all_ciou, all_binary_preds, all_binary_gts = [], [], [], []
    total = 0
    summary_records = []

    total_refined_src1 = 0
    total_kept_src1    = 0
    total_refined_src2 = 0
    total_kept_src2    = 0

    for i, frame_file in enumerate(frame_files, 1):
        vid        = os.path.splitext(frame_file)[0]
        image_path = os.path.join(frame_dir, frame_file)
        audio_path = os.path.join(audio_dir, f"{vid}.wav")

        if not os.path.exists(audio_path) or vid not in gt_dict:
            print(f"[skip] {vid}")
            continue

        print(f"[{i}/{len(frame_files)}] Processing {vid} ...")
        try:
            # ---------- Stage A ----------
            text_A   = run_chat(
                model, processor,
                contents=[
                    {"type": "image", "path": image_path},
                    {"type": "audio", "path": audio_path},
                    {"type": "text",  "text": prompt_locate_A},
                ],
                system_text="You are a multimodal assistant for audio-visual scene understanding (VGGSS Duet - Stage A).",
                do_sample=False, temperature=0.0, top_p=1.0
            )
            obj_A    = parse_duet_locate_json(text_A)
            bboxes_A = obj_A["bboxes"]
            descs_A  = obj_A.get("descriptions", ["", ""])

            img  = Image.open(image_path)
            W, H = img.size
            bboxes_A = [clip_box([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])], W, H) for bb in bboxes_A]

            # ---------- Stage B ----------
            text_B = run_chat(
                model, processor,
                contents=[
                    {"type": "audio", "path": audio_path},
                    {"type": "text",  "text": prompt_audio_B},
                ],
                system_text="You are an audio classification expert (VGGSS Duet - Stage B).",
                do_sample=False, temperature=0.0, top_p=1.0
            )
            obj_B         = parse_duet_audio_json(text_B)
            audio_classes = obj_B["audio_classes"]
            audio_confs   = obj_B["audio_confidence_scores"]

            # ---------- Stage B.5: Anchor Voting (per source, n=7) ----------
            analyses         = []
            av_consistencies = []
            keep_flags       = []

            for k in range(2):
                analysis_k = run_anchor_voting_multi(
                    model, processor, image_path, audio_path,
                    lambda bb=bboxes_A[k], ac=audio_classes[k]: build_prompt_analysis_anchor_voting_no_conf(bb, ac, W, H),
                    n=args.n_votes
                )

                avc    = float(np.clip(float(analysis_k.get("av_consistency", 0.0)), 0.0, 1.0))
                keep_k = bool(analysis_k.get("keep", True))

                # role_tags 정규화
                rtags  = analysis_k.get("role_tags", []) or []
                _clean = []
                for t in rtags:
                    if not isinstance(t, str):
                        continue
                    s = re.sub(r"[\s\-]+", "_", t.strip().lower())[:32]
                    if s and s not in _clean:
                        _clean.append(s)
                rtags = _clean[:4]

                # anchor_votes 정규화 (중복 anchor는 max score)
                avotes = analysis_k.get("anchor_votes", []) or []
                _anchors = []
                for av in avotes:
                    if not isinstance(av, dict):
                        continue
                    name = av.get("anchor", "")
                    sc   = av.get("score", 0.0)
                    if not isinstance(name, str):
                        continue
                    try:
                        sc = float(sc)
                    except Exception:
                        sc = 0.0
                    name = re.sub(r"[\s\-]+", "_", name.strip().lower())[:48]
                    sc   = float(np.clip(sc, 0.0, 1.0))
                    if name:
                        _anchors.append({"anchor": name, "score": sc})
                merged = {}
                for item in _anchors:
                    kname = item["anchor"]
                    merged[kname] = max(merged.get(kname, 0.0), item["score"])
                avotes = sorted([{"anchor": k, "score": v} for k, v in merged.items()], key=lambda x: -x["score"])[:5]

                analyses.append({"av_consistency": avc, "role_tags": rtags, "anchor_votes": avotes, "keep": keep_k})
                av_consistencies.append(avc)
                keep_flags.append(keep_k)

            # ---------- 게이팅 (τ_av=0.75, τ_aud=0.75) ----------
            do_refines = []
            for k in range(2):
                if keep_flags[k] and av_consistencies[k] >= args.tau_av and audio_confs[k] >= args.tau_audio:
                    do_refines.append(False)
                    if k == 0:
                        print(f"  [GATING SRC1] Skipping (keep={keep_flags[k]}, av={av_consistencies[k]:.3f}, audio_conf={audio_confs[k]:.3f})")
                        total_kept_src1 += 1
                    else:
                        print(f"  [GATING SRC2] Skipping (keep={keep_flags[k]}, av={av_consistencies[k]:.3f}, audio_conf={audio_confs[k]:.3f})")
                        total_kept_src2 += 1
                else:
                    do_refines.append(True)
                    if k == 0:
                        print(f"  [GATING SRC1] Proceeding (keep={keep_flags[k]}, av={av_consistencies[k]:.3f}, audio_conf={audio_confs[k]:.3f})")
                        total_refined_src1 += 1
                    else:
                        print(f"  [GATING SRC2] Proceeding (keep={keep_flags[k]}, av={av_consistencies[k]:.3f}, audio_conf={audio_confs[k]:.3f})")
                        total_refined_src2 += 1

            # ---------- Stage C: Refine (per source) ----------
            max_delta_px  = int(max(4, round(min(W, H) * 0.08)))
            bboxes_final  = [bboxes_A[0][:], bboxes_A[1][:]]
            descs_refined = ["", ""]

            for k in range(2):
                if do_refines[k]:
                    prompt_refine = build_prompt_refine(
                        bboxes_A[k], audio_classes[k], W, H,
                        analysis=analyses[k], max_delta_px=max_delta_px
                    )
                    text_C = run_chat(
                        model, processor,
                        contents=[
                            {"type": "image", "path": image_path},
                            {"type": "audio", "path": audio_path},
                            {"type": "text",  "text": prompt_refine},
                        ],
                        system_text=f"You refine bbox for source#{k+1} with class-conditional reasoning (Stage C).",
                        do_sample=False, temperature=0.0, top_p=1.0
                    )
                    try:
                        obj_C        = parse_json_object(text_C)
                        bbox_C       = obj_C.get("bbox", None)
                        changed_flag = obj_C.get("changed", None)
                        ops_obj      = obj_C.get("ops", None)
                        candidate    = bboxes_A[k]

                        if ops_obj:
                            candidate = apply_ops(candidate, ops_obj, W, H, max_delta_px)
                        if isinstance(bbox_C, list) and len(bbox_C) == 4:
                            bbox_C    = clip_box([int(bbox_C[0]), int(bbox_C[1]), int(bbox_C[2]), int(bbox_C[3])], W, H)
                            bbox_C    = clamp_delta(bboxes_A[k], bbox_C, max_delta_px, W, H)
                            candidate = bbox_C
                        if changed_flag is False:
                            bboxes_final[k] = bboxes_A[k]
                        else:
                            bboxes_final[k] = candidate if candidate != bboxes_A[k] else bboxes_A[k]
                        descs_refined[k] = str(obj_C.get("refined_description", "")).strip()
                    except Exception:
                        pass
                else:
                    descs_refined[k] = ""

            # ---------- GT 변환 ----------
            img_cv = cv2.imread(image_path)
            scale_x, scale_y = 224 / W, 224 / H
            gboxes_img, gboxes_224 = [], []

            for b in gt_dict[vid]["boxes_xywh"]:
                if b is None:
                    continue
                g    = clip_box(denorm_xywh_to_xyxy(b, W, H), W, H)
                g224 = clip_box([
                    int(round(g[0] * scale_x)), int(round(g[1] * scale_y)),
                    int(round(g[2] * scale_x)), int(round(g[3] * scale_y)),
                ], 224, 224)
                gboxes_img.append(g)
                gboxes_224.append(g224)

            # ---------- 시각화 (원본) ----------
            if img_cv is not None:
                vis_img = img_cv.copy()
                for pb in bboxes_final:
                    cv2.rectangle(vis_img, (pb[0], pb[1]), (pb[2], pb[3]), (255, 0, 0), 2)
                for gb in gboxes_img:
                    cv2.rectangle(vis_img, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir, f"{vid}.jpg"), vis_img)

            # ---------- 시각화 (224) ----------
            vis_224    = cv2.imread(image_path)
            pboxes_224 = []
            if vis_224 is not None:
                vis_224 = cv2.resize(vis_224, (224, 224), interpolation=cv2.INTER_AREA)
                for pb in bboxes_final:
                    p224 = clip_box([
                        int(round(pb[0] * scale_x)), int(round(pb[1] * scale_y)),
                        int(round(pb[2] * scale_x)), int(round(pb[3] * scale_y)),
                    ], 224, 224)
                    pboxes_224.append(p224)
                    cv2.rectangle(vis_224, (p224[0], p224[1]), (p224[2], p224[3]), (255, 0, 0), 2)
                for gb in gboxes_224:
                    cv2.rectangle(vis_224, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir_224, f"{vid}.jpg"), vis_224)
            else:
                for pb in bboxes_final:
                    pboxes_224.append(clip_box([
                        int(round(pb[0] * scale_x)), int(round(pb[1] * scale_y)),
                        int(round(pb[2] * scale_x)), int(round(pb[3] * scale_y)),
                    ], 224, 224))

            # ---------- Metrics ----------
            pred_mask = np.zeros((224, 224), dtype=np.float32)
            gt_mask   = np.zeros((224, 224), dtype=np.float32)
            for pb in pboxes_224:
                cv2.rectangle(pred_mask, (pb[0], pb[1]), (pb[2], pb[3]), 1, -1)
            for gb in gboxes_224:
                cv2.rectangle(gt_mask, (gb[0], gb[1]), (gb[2], gb[3]), 1, -1)

            ciou_val = evaluator.cal_CIOU(pred_mask, gt_mask, 0.5)
            ap_val   = average_precision_score(gt_mask.flatten(), pred_mask.flatten())

            def _to_xyxy_uint(b):
                return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

            per_source_iou = [None, None]
            matched_mean_iou = None
            if len(gboxes_224) == 2:
                i00, i01 = iou_xyxy(_to_xyxy_uint(pboxes_224[0]), _to_xyxy_uint(gboxes_224[0])), iou_xyxy(_to_xyxy_uint(pboxes_224[0]), _to_xyxy_uint(gboxes_224[1]))
                i10, i11 = iou_xyxy(_to_xyxy_uint(pboxes_224[1]), _to_xyxy_uint(gboxes_224[0])), iou_xyxy(_to_xyxy_uint(pboxes_224[1]), _to_xyxy_uint(gboxes_224[1]))
                if (i00 + i11) >= (i01 + i10):
                    matched_mean_iou = (i00 + i11) / 2.0
                    per_source_iou   = [float(i00), float(i11)]
                else:
                    matched_mean_iou = (i01 + i10) / 2.0
                    per_source_iou   = [float(i01), float(i10)]
            elif len(gboxes_224) == 1:
                i0 = iou_xyxy(_to_xyxy_uint(pboxes_224[0]), _to_xyxy_uint(gboxes_224[0]))
                i1 = iou_xyxy(_to_xyxy_uint(pboxes_224[1]), _to_xyxy_uint(gboxes_224[0]))
                matched_mean_iou = float(max(i0, i1))
                per_source_iou   = [float(i0), None] if i0 >= i1 else [None, float(i1)]

            all_ciou.append(ciou_val)
            all_ap.append(ap_val)
            all_binary_preds.extend(pred_mask.flatten())
            all_binary_gts.extend(gt_mask.flatten())
            total += 1

            bbox_norm_224_src1 = [pboxes_224[0][0]/224, pboxes_224[0][1]/224, pboxes_224[0][2]/224, pboxes_224[0][3]/224]
            bbox_norm_224_src2 = [pboxes_224[1][0]/224, pboxes_224[1][1]/224, pboxes_224[1][2]/224, pboxes_224[1][3]/224]

            per_json = {
                "file_id": vid,
                "bbox_norm_224_src1": bbox_norm_224_src1,
                "bbox_norm_224_src2": bbox_norm_224_src2,
                "descriptions_stageA": descs_A,
                "descriptions_refined": descs_refined,
                "audio_classes": audio_classes,
                "audio_confidences": [round(audio_confs[0], 3), round(audio_confs[1], 3)],
                "analysis": [analyses[0], analyses[1]],
                "gt_classes": gt_dict[vid]["classes"],
                "per_source_iou": per_source_iou,
                "matched_mean_iou": matched_mean_iou,
                "ciou_union@0.5": float(ciou_val),
                "ap_union": float(ap_val),
            }
            with open(os.path.join(bbox_json_dir, f"{vid}.json"), "w") as f:
                json.dump(per_json, f, ensure_ascii=False, indent=2)

            print(f"  src1={bbox_norm_224_src1} | src2={bbox_norm_224_src2} | cIoU@0.5={ciou_val:.3f}, AP={ap_val:.3f}")
            summary_records.append(per_json)

        except Exception as e:
            print(f"  Error @ {vid}: {e}")

    with open(summary_json_path, "w") as f:
        json.dump(summary_records, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY SAVED] {summary_json_path}")

    ciou_arr   = np.array(all_ciou) if all_ciou else np.array([])
    ciou_0_5   = float(np.mean(ciou_arr >= 0.5)) if all_ciou else 0.0
    ciou_0_3   = float(np.mean(ciou_arr >= 0.3)) if all_ciou else 0.0
    thresholds = np.linspace(0, 1, 21)
    results    = [float(np.mean(ciou_arr >= t)) for t in thresholds] if all_ciou else [0.0] * len(thresholds)
    auc_       = float(metrics.auc(thresholds, results)) if all_ciou else 0.0
    cap        = float(np.mean(all_ap)) if all_ap else 0.0
    piap       = float(average_precision_score(all_binary_gts, all_binary_preds)) if all_binary_gts else 0.0

    print("=" * 50)
    print(f"Total evaluated: {total}")
    print(f"cIoU@0.3: {ciou_0_3:.3f}")
    print(f"cIoU@0.5: {ciou_0_5:.3f}")
    print(f"AUC: {auc_:.3f}")
    print(f"CAP: {cap:.3f}")
    print(f"PIAP: {piap:.3f}")
    print("=" * 50)
    print(f"GATING STATISTICS:")
    print(f"Source 1 - Refined: {total_refined_src1}/{total} ({total_refined_src1/total*100:.1f}%), Kept: {total_kept_src1}/{total} ({total_kept_src1/total*100:.1f}%)")
    print(f"Source 2 - Refined: {total_refined_src2}/{total} ({total_refined_src2/total*100:.1f}%), Kept: {total_kept_src2}/{total} ({total_kept_src2/total*100:.1f}%)")
    print("=" * 50)