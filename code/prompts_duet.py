"""
듀엣(duet) 스크립트용 프롬프트 빌더
- GAR_music_duet.py
- GAR_vggss_duet.py

참고:
- build_prompt_locate_duet / build_prompt_audio_duet: dataset_name 파라미터로 구분
- build_prompt_analysis_anchor_voting: audio_conf_score 포함 (music_duet용)
- build_prompt_analysis_anchor_voting_no_conf: audio_conf_score 미포함 (vggss_duet용)
- build_prompt_refine: 단일 소스 박스 정제 (duet 공용)
"""


def build_prompt_locate_duet(dataset_name="MUSIC DUET"):
    """Stage A (DUET): IMAGE + AUDIO -> 2 bboxes + 2 descriptions"""
    return (
        "You are an assistant for audio-visual sound source localization.\n"
        f"TASK ({dataset_name}): From the given IMAGE and AUDIO of a DUET scene, "
        "return the TWO regions in the image where EACH of the TWO sound sources comes from.\n\n"
        "STRICT OUTPUT:\n"
        "Return ONLY a raw JSON object with EXACTLY these fields:\n"
        "{\n"
        '  "bboxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]],\n'
        '  "descriptions": ["<object emitting sound #1>", "<object emitting sound #2>"]\n'
        "}\n"
        "- Coordinates must be 4 INTEGERS per bbox in pixel coordinates of the ORIGINAL image (x1<x2, y1<y2).\n"
        "- Return EXACTLY TWO boxes for the two main sound-emitting objects. Do NOT return more or less.\n"
        "- Do not add extra fields or any extra text outside the JSON."
    )


def build_prompt_audio_duet(dataset_name="MUSIC DUET"):
    """Stage B (DUET): AUDIO only -> 2 audio_class + 2 audio_confidence_score"""
    return (
        "You are an audio classification expert.\n\n"
        f"TASK ({dataset_name}): Listen to the AUDIO and classify the TWO dominant audio event classes in this duet.\n"
        "Return ONLY a raw JSON object with EXACTLY these fields:\n"
        "{\n"
        '  "audio_classes": ["<class #1>", "<class #2>"],\n'
        '  "audio_confidence_scores": [c1, c2]\n'
        "}\n"
        '- Each class must be a short, lowercase label (e.g., "violin", "piano", "drum set").\n'
        "- Each confidence must be a float in [0.0, 1.0].\n"
        '- If uncertain, still output two items (use "unknown" with a low confidence).\n'
        "- Do not add any extra fields or text."
    )


def build_prompt_analysis_anchor_voting(prev_bbox, audio_class, audio_conf_score, W, H):
    """
    Stage B.5: 점검(Anchor Voting) - per source, audio_conf_score 포함 (music_duet용)
    """
    try:
        conf_str = f"{float(audio_conf_score):.3f}"
    except Exception:
        conf_str = "0.000"

    return f"""
You will analyze THIS SPECIFIC sample to verify whether the AUDIO-indicated sound source is actually VISIBLE in the IMAGE.
Your output must be inferred ONLY from THIS image+audio. Do not assume unseen parts.

CONTEXT:
- previous_bbox (candidate region to check proximity): {prev_bbox} within image [{W}x{H}]
- audio_class: "{audio_class}"
- audio_confidence_score: {conf_str}

DEFINITIONS:
- "anchor_votes": propose 0~5 concise, lowercase anchors that represent the visible cause of the sound indicated by the audio_class.
  Examples:
    audio_class="applause"    -> anchors like "hands_clapping"
    audio_class="violin"      -> anchors like "bow_on_strings", "left_hand_fret", "violin_body"
    audio_class="drum set"    -> anchors like "drumstick_contact", "snare_surface", "hi_hat_edge"
    audio_class="dog barking" -> anchors like "dog_mouth_open"
  Each item must be: {{"anchor":"<token_with_underscores>","score": s}} where s∈[0,1] reflects visual evidence in THIS image.
  If the anchor is NOT clearly visible, either omit it or assign a very low score.
- "role_tags": up to 4 short, lowercase tokens summarizing the visible roles/parts you relied on (free-form; avoid redundancy).
- "av_consistency" ∈ [0,1] must jointly reflect:
    (i) alignment between audio_class and visible evidence,
    (ii) whether the key evidence lies INSIDE or NEAR the previous_bbox (closer → higher),
    (iii) overall clarity (avoid reusing constant values).
- "keep": true ONLY if refinement can be safely skipped based on a holistic judgment combining
    av_consistency, audio_confidence_score, and whether strong anchors lie INSIDE/NEAR previous_bbox.

STRICT OUTPUT:
Return ONLY a raw JSON object with EXACTLY these fields:
{{
  "av_consistency": <float 0..1>,
  "role_tags": [<0..4 short tokens>],
  "anchor_votes": [{{"anchor":"<token_with_underscores>","score":<0..1>}}, ...],
  "keep": <true|false>
}}
""".strip()


def build_prompt_analysis_anchor_voting_no_conf(prev_bbox, audio_class, W, H):
    """
    Stage B.5: 분석(Anchor Voting) - per source, audio_conf_score 미포함 (vggss_duet용)
    """
    return f"""
You will analyze THIS SPECIFIC sample between a previous bbox and the AUDIO class.
Your output must be inferred ONLY from THIS image+audio.

CONTEXT:
- previous_bbox: {prev_bbox} within image size [{W}x{H}]
- audio_class: "{audio_class}"

ROLE TAGS (open set):
- Choose 0~4 short, free-form tags that describe visible, semantically relevant parts/roles for THIS audio_class in THIS image.
- Each tag should be lowercase and 1~3 words; prefer snake_case (e.g., "bow_hand", "mouthpiece_contact", "drum_surface").
- If nothing is clearly visible/relevant, return [] (DO NOT default to any tag).

TASK:
1) Propose 0~5 semantic anchors (parts) relevant to the audio_class that are actually suggested by THIS sample.
   Each item: {{"anchor":"<name>","score":s}} with s in [0,1].
   If no anchors are clearly visible, return [].
2) role_tags: choose 0~4 free-form tags that are actually visible/relevant in THIS image.
   If nothing is clear, return [].
3) av_consistency in [0,1] must reflect THIS sample's A/V agreement.
   Do NOT reuse a constant value; compute based on THIS sample's anchors/visibility.
4) keep=true ONLY if refinement can be safely skipped.

STRICT OUTPUT:
Return ONLY a raw JSON object with EXACTLY these fields:
{{
  "av_consistency": <float 0..1>,
  "role_tags": [<0..4 short strings>],
  "anchor_votes": [{{"anchor":"<name>","score":<0..1>}}, ...],
  "keep": <true|false>
}}
""".strip()


def build_prompt_refine(prev_bbox, audio_class, W, H, analysis=None, max_delta_px=12):
    """Stage C: Refine - per source (duet 공용)"""
    analysis = analysis or {}
    return f"""
You are refining a bounding box for ONE sound-emitting object, using IMAGE and AUDIO, and integrating prior outputs.

CONTEXT:
- previous_bbox (Stage A): {prev_bbox}
- audio_class (Stage B): "{audio_class}"
- image_size: width={W}, height={H}
- analysis.av_consistency: {analysis.get('av_consistency', 0.0)}
- analysis.role_tags: {analysis.get('role_tags', [])}
- analysis.anchor_votes: {analysis.get('anchor_votes', [])}
- analysis.keep: {analysis.get('keep', False)}

REFINE RULES:
1) Make ONE final bbox that best matches the audio_class and verified anchors; prefer minimal changes from previous_bbox.
2) Keep the box inside [0,{W-1}]x[0,{H-1}] with x1<x2, y1<y2 (all INTEGERS).
3) Be conservative: limit total coordinate change within ±{max_delta_px} px on each side unless previous box is clearly wrong.
4) You may return the SAME box if re-check suggests no change.
5) If appropriate, describe the procedural operation via "ops":
   - ops.type ∈ ["delta","expand","shrink","recenter"]
   - "delta": one-side micro-adjust using any of {{"dx","dy","dl","dr","dt","db"}} (integers, pixels)
   - "expand"/"shrink": integer "amount" applied symmetrically to all four sides
   - "recenter": integer-pixel "target": [cx, cy] as the new center (preserve size)
6) Provide "refined_description" (2~4 sentences) explaining the scene, how the sound (mention audio_class) relates to visible objects, without speculation.

STRICT OUTPUT:
Return ONLY a raw JSON object:
{{
  "bbox": [x1, y1, x2, y2],
  "changed": true/false,
  "ops": {{"type":"delta|expand|shrink|recenter", ...}} | null,
  "refined_description": "a concise multi-sentence description about the scene and the sound source"
}}
""".strip()
