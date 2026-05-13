import re
import json
import numpy as np
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor


def load_qwen_omni_thinker(model_id="Qwen/Qwen2.5-Omni-7B"):
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return model, processor


@torch.inference_mode()
def run_chat(model, processor, contents, system_text="You are a multimodal assistant.",
             do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=256):
    model_device = next(model.parameters()).device
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": contents},
    ]
    inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, padding=True
    ).to(model_device)
    out_ids = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens
    )
    text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    return text.strip()


def _strip_assistant_header(t: str) -> str:
    for mk in ["\nassistant\n", "\nassistant:", "assistant\n", "assistant:"]:
        if mk in t:
            t = t.split(mk)[-1]
    return t.strip()


def parse_json_object(text: str):
    t = _strip_assistant_header(text)
    t = t.replace("```json", "```").replace("```JSON", "```")
    if "```" in t:
        parts = t.split("```")
        for i in range(1, len(parts), 2):
            chunk = parts[i].strip()
            try:
                return json.loads(chunk)
            except Exception:
                pass
        t = "".join([parts[0]] + parts[2::2])
    for m in re.finditer(r"({[\s\S]*?})", t):
        try:
            return json.loads(m.group(1))
        except Exception:
            continue
    raise ValueError("No valid JSON object found.")


def parse_duet_locate_json(text: str):
    obj = parse_json_object(text)
    bboxes = obj.get("bboxes")
    descs  = obj.get("descriptions", ["", ""])
    if not (isinstance(bboxes, list) and len(bboxes) == 2
            and all(isinstance(bb, list) and len(bb) == 4 for bb in bboxes)):
        raise ValueError("Stage A (duet): need exactly 2 bboxes.")
    bboxes = [[int(x) for x in bb] for bb in bboxes]
    if not (isinstance(descs, list) and len(descs) == 2):
        descs = ["", ""]
    return {"bboxes": bboxes, "descriptions": descs}


def parse_duet_audio_json(text: str):
    obj = parse_json_object(text)
    classes = obj.get("audio_classes", ["unknown", "unknown"])
    confs   = obj.get("audio_confidence_scores", [0.0, 0.0])
    if not (isinstance(classes, list) and len(classes) == 2):
        classes = ["unknown", "unknown"]
    if not (isinstance(confs, list) and len(confs) == 2):
        confs = [0.0, 0.0]
    classes = [str(c).strip().lower() if c else "unknown" for c in classes]
    confs = [float(x) if isinstance(x, (int, float)) else 0.0 for x in confs]
    confs = [float(np.clip(c, 0.0, 1.0)) for c in confs]
    return {"audio_classes": classes, "audio_confidence_scores": confs}


def run_anchor_voting_multi(model, processor, image_path, audio_path, prompt_builder, n=7):
    """
    Anchor-Voting을 n회 반복 실행하고 결과를 합의(average/majority)로 결합.
    - av_consistency: 평균
    - role_tags: 빈도 상위 4개
    - anchor_votes: 동일 anchor 이름별 score 평균
    - keep: 과반(majority) 이상 True면 True
    """
    outs = []
    for _ in range(n):
        t = run_chat(
            model, processor,
            contents=[
                {"type": "image", "path": image_path},
                {"type": "audio", "path": audio_path},
                {"type": "text",  "text": prompt_builder()},
            ],
            system_text="You verify visible evidence for the audio-indicated source and judge proximity to the previous bbox (Stage B.5).",
            do_sample=True, temperature=0.4, top_p=0.9, max_new_tokens=256
        )
        try:
            outs.append(parse_json_object(t))
        except Exception:
            pass

    if not outs:
        return {"av_consistency": 0.0, "role_tags": [], "anchor_votes": [], "keep": False}

    # av_consistency 평균
    av_list = []
    for o in outs:
        try:
            av_list.append(float(o.get("av_consistency", 0.0)))
        except Exception:
            av_list.append(0.0)
    av_consistency = float(np.clip(np.mean(av_list) if av_list else 0.0, 0.0, 1.0))

    # role_tags 빈도 상위 4개
    freq = {}
    for o in outs:
        tags = o.get("role_tags") or []
        if isinstance(tags, list):
            for tg in tags:
                if isinstance(tg, str):
                    tok = re.sub(r"[\s\-]+", "_", tg.strip().lower())
                    if tok:
                        freq[tok] = freq.get(tok, 0) + 1
    role_tags = [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:4]]

    # anchor_votes: anchor별 score 평균
    agg = {}
    for o in outs:
        avs = o.get("anchor_votes") or []
        if isinstance(avs, list):
            for it in avs:
                if not isinstance(it, dict):
                    continue
                name = it.get("anchor", "")
                try:
                    sc = float(it.get("score", 0.0))
                except Exception:
                    sc = 0.0
                name = re.sub(r"[\s\-]+", "_", str(name).strip().lower())[:48]
                if not name:
                    continue
                agg.setdefault(name, []).append(float(np.clip(sc, 0.0, 1.0)))
    anchor_votes = [{"anchor": k, "score": float(np.mean(v))} for k, v in agg.items()]
    anchor_votes.sort(key=lambda x: -x["score"])

    # keep: 과반 이상 True
    keep_votes = sum(1 for o in outs if bool(o.get("keep", False)))
    keep = keep_votes >= (len(outs) // 2 + 1)

    return {
        "av_consistency": av_consistency,
        "role_tags": role_tags,
        "anchor_votes": anchor_votes,
        "keep": keep,
    }
