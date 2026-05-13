"""
단일 소스(solo/single) 스크립트용 프롬프트 빌더
- GAR_music_solo.py
- GAR_vggss_single.py
"""


def build_prompt_locate():
    """Stage A: IMAGE + AUDIO -> bbox + description"""
    return """
You are an assistant for audio-visual sound source localization.

TASK (Stage A):
Given an IMAGE and an AUDIO clip from the same scene:
1) Find exactly ONE bounding box [x1, y1, x2, y2] for the MAIN sound-emitting object in the IMAGE.
2) Provide a concise visual description of the sound-emitting object.

STRICT OUTPUT:
Return ONLY a raw JSON object with EXACTLY these fields:
{
  "bbox": [x1, y1, x2, y2],
  "description": "visual description of the sound-emitting object"
}
- bbox must be 4 INTEGERS in pixel coordinates of the ORIGINAL image (x1<x2, y1<y2).
- Do not add extra fields or any extra text outside the JSON.
"""


def build_prompt_audio_only():
    """Stage B: AUDIO only -> audio_class + audio_confidence_score"""
    return """
You are an audio classification expert.

TASK (Stage B):
Listen to the AUDIO and classify the dominant audio event as a short, lowercase class name
(e.g., "violin", "piano", "dog barking", "engine", "drum set").
Also provide a confidence score in [0.0, 1.0].

STRICT OUTPUT:
Return ONLY a raw JSON object with EXACTLY these fields:
{
  "audio_class": "<concise class name>",
  "audio_confidence_score": 0.0
}
- "audio_class" must be short and lowercase.
- "audio_confidence_score" must be a float in [0.0, 1.0].
- Do not add extra fields or any extra text outside the JSON.
"""


def build_prompt_analysis_anchor_voting(prev_bbox, audio_class, audio_conf_score, W, H):
    """Stage B.5: 점검(Anchor Voting)"""
    try:
        audio_conf_str = f"{float(audio_conf_score):.3f}"
    except Exception:
        audio_conf_str = "0.000"

    return f"""
You will analyze THIS SPECIFIC sample to verify whether the AUDIO-indicated sound source is actually VISIBLE in the IMAGE.
Your output must be inferred ONLY from THIS image+audio. Do not assume unseen parts.

CONTEXT:
- previous_bbox (candidate region to check proximity): {prev_bbox} within image [{W}x{H}]
- audio_class: "{audio_class}"
- audio_confidence_score: {audio_conf_str}

DEFINITIONS:
- "anchor_votes": propose 0~5 concise, lowercase anchors that represent the visible cause of the sound indicated by the audio_class.
  Examples:
    audio_class="applause"  -> anchors like "hands_clapping"
    audio_class="violin"    -> anchors like "bow_on_strings", "left_hand_fret", "violin_body"
    audio_class="drum set"  -> anchors like "drumstick_contact", "snare_surface", "hi_hat_edge"
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
"""


def build_prompt_refine(prev_bbox, audio_class, W, H, analysis=None, max_delta_px=12):
    """Stage C: Refine (ops + clamp)"""
    analysis = analysis or {}
    return f"""
You are refining a bounding box for the MAIN sound-emitting object, using IMAGE and AUDIO, and integrating prior outputs.

CONTEXT:
- previous_bbox (from Stage A): {prev_bbox}
- audio_class (from Stage B): "{audio_class}"
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
6) Provide "refined_description" (2–4 sentences) explaining the scene, how the sound (mention audio_class) relates to visible objects, without speculation.

STRICT OUTPUT:
Return ONLY a raw JSON object:
{{
  "bbox": [x1, y1, x2, y2],
  "changed": true/false,
  "ops": {{"type":"delta|expand|shrink|recenter", ...}} | null,
  "refined_description": "a concise multi-sentence description about the scene and the sound source"
}}
"""
