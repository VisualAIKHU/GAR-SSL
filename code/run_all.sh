#!/bin/bash
# =============================================================================
# GAR_CVPR26 - Run All Datasets
# Usage:
#   bash run_all.sh                        # 전체 실행 (순차), N_VOTES=5 (기본값)
#   bash run_all.sh music_solo             # 특정 데이터셋만 실행
#   bash run_all.sh music_duet
#   bash run_all.sh vggss_single
#   bash run_all.sh vggss_duet
#   bash run_all.sh all 5                  # 전체 실행, N_VOTES=5
#   bash run_all.sh music_solo 3           # music_solo만 실행, N_VOTES=3
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────
# 공통 설정
# ─────────────────────────────────────────────
MODEL_ID="Qwen/Qwen2.5-Omni-7B"
N_VOTES="${2:-5}"

# ─────────────────────────────────────────────
# 데이터셋별 설정
# ─────────────────────────────────────────────

run_music_solo() {
    echo "=========================================="
    echo "  [1/4] MUSIC Solo"
    echo "=========================================="
    python GAR_music_solo.py \
        --model_id    "$MODEL_ID" \
        --frame_dir   "/data/subin/MUSIC/solo/test/frames" \
        --audio_dir   "/data/subin/MUSIC/solo/test/audio" \
        --gt_path     "/data/subin/metadata/music_solo.json" \
        --out_root    "/data/subin/GAR_CVPR26/outputs/GAR_music_solo" \
        --cuda_device "0" \
        --n_votes     $N_VOTES \
        --tau_av      0.75 \
        --tau_audio   0.75
}

run_music_duet() {
    echo "=========================================="
    echo "  [2/4] MUSIC Duet"
    echo "=========================================="
    python GAR_music_duet.py \
        --model_id    "$MODEL_ID" \
        --frame_dir   "/data/subin/MUSIC/duet/test/frames" \
        --audio_dir   "/data/subin/MUSIC/duet/test/audio" \
        --gt_path     "/data/subin/metadata/music_duet.json" \
        --out_root    "/data/subin/GAR_CVPR26/outputs/GAR_music_duet" \
        --cuda_device "1" \
        --n_votes     $N_VOTES \
        --tau_av      0.75 \
        --tau_audio   0.50
}

run_vggss_single() {
    echo "=========================================="
    echo "  [3/4] VGGSound Single"
    echo "=========================================="
    python GAR_vggss_single.py \
        --model_id    "$MODEL_ID" \
        --frame_dir   "/data/subin/VGGSound/test/frames" \
        --audio_dir   "/data/subin/VGGSound/test/audio" \
        --gt_path     "/data/subin/metadata/vggss.json" \
        --out_root    "/data/subin/GAR_CVPR26/outputs/GAR_vggss_single" \
        --cuda_device "0" \
        --n_votes     $N_VOTES \
        --tau_av      0.50 \
        --tau_audio   0.50
}

run_vggss_duet() {
    echo "=========================================="
    echo "  [4/4] VGGSound Duet"
    echo "=========================================="
    python GAR_vggss_duet.py \
        --model_id    "$MODEL_ID" \
        --frame_dir   "/data/subin/VGGSound_duet/test/frames" \
        --audio_dir   "/data/subin/VGGSound_duet/test/audio" \
        --gt_path     "/data/subin/VGGSound_duet/vggss_duet_test.json" \
        --out_root    "/data/subin/GAR_CVPR26/outputs/GAR_vggss_duet" \
        --cuda_device "0" \
        --n_votes     $N_VOTES \
        --tau_av      0.75 \
        --tau_audio   0.75
}

# ─────────────────────────────────────────────
# 실행 분기
# ─────────────────────────────────────────────
TARGET="${1:-all}"

echo "N_VOTES = $N_VOTES"

case "$TARGET" in
    music_solo)    run_music_solo ;;
    music_duet)    run_music_duet ;;
    vggss_single)  run_vggss_single ;;
    vggss_duet)    run_vggss_duet ;;
    all)
        run_music_solo
        run_music_duet
        run_vggss_single
        run_vggss_duet
        echo "=========================================="
        echo "  All datasets finished."
        echo "=========================================="
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash run_all.sh [music_solo|music_duet|vggss_single|vggss_duet|all]"
        exit 1
        ;;
esac