import os
import json
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn import metrics

from model_utils import load_qwen_omni_thinker, run_chat, parse_json_object, run_anchor_voting_multi
from bbox_utils import clip_box, clamp_delta, apply_ops, denorm_xyxy_to_xyxy
from evaluator import Evaluator
from data_utils import load_vggss_gt
from prompts_single import (
    build_prompt_locate,
    build_prompt_audio_only,
    build_prompt_analysis_anchor_voting,
    build_prompt_refine,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAR - VGGSound Single")
    parser.add_argument("--model_id",     type=str,   default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--frame_dir",    type=str,   default="/data/subin/VGGSound/test/frames")
    parser.add_argument("--audio_dir",    type=str,   default="/data/subin/VGGSound/test/audio")
    parser.add_argument("--gt_path",      type=str,   default="/data/subin/metadata/vggss.json")
    parser.add_argument("--out_root",     type=str,   default="/data/subin/GAR_CVPR26/outputs/GAR_vggss_single")
    parser.add_argument("--cuda_device",  type=str,   default="0")
    parser.add_argument("--n_votes",      type=int,   default=7)
    parser.add_argument("--tau_av",       type=float, default=0.5)
    parser.add_argument("--tau_audio",    type=float, default=0.5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    model, processor = load_qwen_omni_thinker(args.model_id)

    prompt_locate_A = build_prompt_locate()
    prompt_audio_B  = build_prompt_audio_only()

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

    gt_dict     = load_vggss_gt(gt_path)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    print(f"Found {len(frame_files)} frames")

    evaluator = Evaluator()
    all_ap, all_ciou, all_binary_preds, all_binary_gts = [], [], [], []
    total, total_refined, total_kept = 0, 0, 0
    summary_records = []

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
            text_A = run_chat(
                model, processor,
                contents=[
                    {"type": "image", "path": image_path},
                    {"type": "audio", "path": audio_path},
                    {"type": "text",  "text": prompt_locate_A},
                ],
                system_text="You are a multimodal assistant for audio-visual scene understanding (Stage A).",
                do_sample=False, temperature=0.0, top_p=1.0
            )
            obj_A  = parse_json_object(text_A)
            bbox_A = obj_A.get("bbox", None)
            desc_A = str(obj_A.get("description", "")).strip()
            if not (isinstance(bbox_A, list) and len(bbox_A) == 4):
                raise ValueError("Stage A: invalid bbox")

            img  = Image.open(image_path)
            W, H = img.size
            bbox_A = clip_box([int(bbox_A[0]), int(bbox_A[1]), int(bbox_A[2]), int(bbox_A[3])], W, H)

            # ---------- Stage B ----------
            text_B = run_chat(
                model, processor,
                contents=[
                    {"type": "audio", "path": audio_path},
                    {"type": "text",  "text": prompt_audio_B},
                ],
                system_text="You are an audio classification expert (Stage B).",
                do_sample=False, temperature=0.0, top_p=1.0
            )
            obj_B = parse_json_object(text_B)
            audio_class = str(obj_B.get("audio_class", "")).strip().lower() or "unknown"
            try:
                audio_conf_score = float(obj_B.get("audio_confidence_score", 0.0))
            except Exception:
                audio_conf_score = 0.0

            # ---------- Stage B.5: Anchor Voting ----------
            analysis = run_anchor_voting_multi(
                model, processor, image_path, audio_path,
                lambda: build_prompt_analysis_anchor_voting(bbox_A, audio_class, audio_conf_score, W, H),
                n=args.n_votes
            )
            av_consistency = float(np.clip(analysis.get("av_consistency", 0.0), 0.0, 1.0))
            role_tags    = analysis.get("role_tags", []) or []
            anchor_votes = analysis.get("anchor_votes", []) or []
            keep_flag    = bool(analysis.get("keep", False))

            # ---------- 게이팅 ----------
            if keep_flag and av_consistency >= args.tau_av and audio_conf_score >= args.tau_audio:
                do_refine = False
                total_kept += 1
                print(f"  [GATING] Skipping refinement (keep={keep_flag}, av={av_consistency:.3f}, audio_conf={audio_conf_score:.3f})")
            else:
                do_refine = True
                total_refined += 1
                print(f"  [GATING] Proceeding to refinement (keep={keep_flag}, av={av_consistency:.3f}, audio_conf={audio_conf_score:.3f})")

            # ---------- Stage C: Refine ----------
            max_delta_px = int(max(4, round(min(W, H) * 0.08)))
            bbox_final   = bbox_A
            desc_refined = ""

            if do_refine:
                prompt_refine = build_prompt_refine(
                    bbox_A, audio_class, W, H,
                    analysis={"av_consistency": av_consistency, "role_tags": role_tags,
                              "anchor_votes": anchor_votes, "keep": keep_flag},
                    max_delta_px=max_delta_px
                )
                text_C = run_chat(
                    model, processor,
                    contents=[
                        {"type": "image", "path": image_path},
                        {"type": "audio", "path": audio_path},
                        {"type": "text",  "text": prompt_refine},
                    ],
                    system_text="You refine bounding boxes with class-conditional reasoning (Stage C).",
                    do_sample=False, temperature=0.0, top_p=1.0
                )
                try:
                    obj_C        = parse_json_object(text_C)
                    bbox_C       = obj_C.get("bbox", None)
                    changed_flag = obj_C.get("changed", None)
                    ops_obj      = obj_C.get("ops", None)
                    candidate    = bbox_A

                    if ops_obj:
                        candidate = apply_ops(candidate, ops_obj, W, H, max_delta_px)
                    if isinstance(bbox_C, list) and len(bbox_C) == 4:
                        bbox_C    = clip_box([int(bbox_C[0]), int(bbox_C[1]), int(bbox_C[2]), int(bbox_C[3])], W, H)
                        bbox_C    = clamp_delta(bbox_A, bbox_C, max_delta_px, W, H)
                        candidate = bbox_C
                    if changed_flag is False:
                        bbox_final = bbox_A
                    else:
                        bbox_final = candidate if candidate != bbox_A else bbox_A
                    desc_refined = str(obj_C.get("refined_description", "")).strip()
                except Exception:
                    pass

            # ---------- 시각화/저장 ----------
            gt_box_xyxy = clip_box(denorm_xyxy_to_xyxy(gt_dict[vid], W, H), W, H)

            vis_img = cv2.imread(image_path)
            if vis_img is not None:
                cv2.rectangle(vis_img, (bbox_final[0], bbox_final[1]), (bbox_final[2], bbox_final[3]), (255, 0, 0), 2)
                cv2.rectangle(vis_img, (gt_box_xyxy[0], gt_box_xyxy[1]), (gt_box_xyxy[2], gt_box_xyxy[3]), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir, f"{vid}.jpg"), vis_img)

            scale_x, scale_y = 224 / W, 224 / H
            pbox_224 = clip_box([
                int(round(bbox_final[0] * scale_x)), int(round(bbox_final[1] * scale_y)),
                int(round(bbox_final[2] * scale_x)), int(round(bbox_final[3] * scale_y)),
            ], 224, 224)
            gbox_224 = clip_box([
                int(round(gt_box_xyxy[0] * scale_x)), int(round(gt_box_xyxy[1] * scale_y)),
                int(round(gt_box_xyxy[2] * scale_x)), int(round(gt_box_xyxy[3] * scale_y)),
            ], 224, 224)

            vis_224 = cv2.imread(image_path)
            if vis_224 is not None:
                vis_224 = cv2.resize(vis_224, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.rectangle(vis_224, (pbox_224[0], pbox_224[1]), (pbox_224[2], pbox_224[3]), (255, 0, 0), 2)
                cv2.rectangle(vis_224, (gbox_224[0], gbox_224[1]), (gbox_224[2], gbox_224[3]), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir_224, f"{vid}.jpg"), vis_224)

            bbox_norm_224 = [pbox_224[0]/224, pbox_224[1]/224, pbox_224[2]/224, pbox_224[3]/224]
            per_json = {
                "file_id": vid,
                "bbox_norm_224": bbox_norm_224,
                "description_stageA": desc_A,
                "description_refined": desc_refined,
                "audio_class": audio_class,
                "audio_confidence": round(audio_conf_score, 3),
                "analysis": {
                    "av_consistency": round(av_consistency, 3),
                    "role_tags": role_tags,
                    "anchor_votes": anchor_votes,
                    "keep": keep_flag,
                }
            }
            with open(os.path.join(bbox_json_dir, f"{vid}.json"), "w") as f:
                json.dump(per_json, f, ensure_ascii=False, indent=2)

            # ----- Metrics -----
            pred_mask = np.zeros((224, 224), dtype=np.float32)
            gt_mask   = np.zeros((224, 224), dtype=np.float32)
            cv2.rectangle(pred_mask, (pbox_224[0], pbox_224[1]), (pbox_224[2], pbox_224[3]), 1, -1)
            cv2.rectangle(gt_mask,   (gbox_224[0], gbox_224[1]), (gbox_224[2], gbox_224[3]), 1, -1)

            ciou_val = evaluator.cal_CIOU(pred_mask, gt_mask, 0.5)
            ap_val   = average_precision_score(gt_mask.flatten(), pred_mask.flatten())
            print(f"  FINAL bbox_norm_224 = {bbox_norm_224} | cIoU={ciou_val:.3f}, AP={ap_val:.3f}")

            all_ciou.append(ciou_val)
            all_ap.append(ap_val)
            all_binary_preds.extend(pred_mask.flatten())
            all_binary_gts.extend(gt_mask.flatten())
            total += 1
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
    print(f"AP: {cap:.3f}")
    print(f"PIAP: {piap:.3f}")
    print("=" * 50)
    print(f"GATING STATISTICS:")
    print(f"Refinement executed: {total_refined}/{total} ({total_refined/total*100:.1f}%)")
    print(f"Refinement skipped:  {total_kept}/{total} ({total_kept/total*100:.1f}%)")
    print("=" * 50)