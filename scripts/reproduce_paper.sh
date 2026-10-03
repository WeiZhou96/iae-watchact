#!/usr/bin/env bash
# Every command used for the results of the paper, in order. The run names are the ones read by
# scripts/analysis/export_release.py, which writes the files of release/.
#
# Environments (see README): main = Python 3.11 + PyTorch + Transformers; hands = MediaPipe; pose = rtmlib;
# vlm = Transformers with FP8 support. Install this package in each of them (pip install -e .).
# GPU stages take a shard index and the number of shards; run several shards in parallel with
# CUDA_VISIBLE_DEVICES to use more GPUs. Paths come from configs/paths.yaml or the IAE_* variables.
set -euo pipefail
cd "$(dirname "$0")/.."
D=$(python -c "from iae.config import DATA_ROOT; print(DATA_ROOT)")
OUT=$(python -c "from iae.config import OUT_ROOT; print(OUT_ROOT)")
SEEDS="0 1 2"
MAIN="--struct 0.1 --struct-epochs 20 --fixed-cost 0.0 --aug pair"

# ---------------------------------------------------------------- 1. Perception (all 1,255 requests)
python scripts/pipeline/s0_import_watchact.py          # request IDs (release/requests.json keys)
python scripts/pipeline/s1_frames.py 16                 # 10-fps frames
python scripts/pipeline/s2_detect_final.py              # Grounding DINO on the anchor frame (GPU)
python scripts/pipeline/s3_register.py                  # registration to the public layout
python scripts/pipeline/s4_track.py 0 1                 # SAM 2.1 propagation from the anchor frame (GPU)
python scripts/pipeline/s5a_person.py 0 1               # person boxes (GPU)
python scripts/pipeline/s5b_hands.py 16                 # hands environment: MediaPipe Hands
python scripts/pipeline/s5c_body.py 16                  # pose environment: RTMPose-m
python scripts/train_eval/build_cache.py feat_cache_v3 tracks 0 1

# ---------------------------------------------------------------- 2. IAE (Tables 1 and 2)
for s in $SEEDS; do
  python scripts/train_eval/s8_cv.py --run abl_notemporal_s$s --seed $s $MAIN --cache feat_cache_v3
done
python scripts/train_eval/ensemble.py ens_final 0 abl_notemporal_s0 abl_notemporal_s1 abl_notemporal_s2
python scripts/train_eval/s11_episodic.py episodic_v1 v1                  # Table 4 (no learning)

# ---------------------------------------------------------------- 3. Ablations and controls (Table 3)
python scripts/train_eval/derive_noray_cache.py feat_cache_v3 feat_cache_v2   # first 22 features, no rays
python scripts/train_eval/s13_forward_track.py 0 1                         # forward-tracking control (GPU)
python scripts/train_eval/build_cache.py feat_cache_v3_fwd tracks_fwd 0 1
python scripts/train_eval/s14_track_subset.py 32 0 1                       # 32-frame pipeline (GPU)
python scripts/train_eval/build_cache.py feat_cache_v3_s32 tracks_s32 0 1
for s in $SEEDS; do
  python scripts/train_eval/s8_cv.py --run abl_final_s$s       --seed $s $MAIN --cache feat_cache_v2          # - arm/head rays
  python scripts/train_eval/s8_cv.py --run abl_full_s$s        --seed $s $MAIN --cache feat_cache_v3 --arch temporal
  python scripts/train_eval/s8_cv.py --run abl_m_nostruct_s$s  --seed $s --fixed-cost 0.0 --cache feat_cache_v3   # - program loss
  python scripts/train_eval/s8_cv.py --run rev_ham_s$s         --seed $s --struct 0.1 --struct-epochs 20 --fixed-cost 0.0 --aug hamming --cache feat_cache_v3
  python scripts/train_eval/s8_cv.py --run abl_m_nogrammar_s$s --seed $s $MAIN --cache feat_cache_v3 --nc-program threshold
  python scripts/train_eval/s8_cv.py --run abl_m_notrack_s$s   --seed $s $MAIN --cache feat_cache_v3_static   # final-frame boxes only
  python scripts/train_eval/s8_cv.py --run rev_fwd_s$s         --seed $s $MAIN --cache feat_cache_v3_fwd
  python scripts/train_eval/s8_cv.py --run rev_sub32_s$s       --seed $s $MAIN --cache feat_cache_v3 --subsample 32
  python scripts/train_eval/s8_cv.py --run rev_s32full_s$s     --seed $s $MAIN --cache feat_cache_v3_s32
  python scripts/train_eval/s8_cv.py --run abl_m_blocked_s$s   --seed $s $MAIN --cache feat_cache_v3 --folds blocked
  python scripts/train_eval/s12_relation.py --run rel_s$s --seed $s                                           # relation model
  python scripts/train_eval/s15_nested_lambda.py --seed $s --main abl_notemporal                              # nested choice of lambda
done
python scripts/baselines/baseline_rules.py rule_heuristic feat_cache_v3                                       # - learning
python scripts/train_eval/ensemble.py rule_heuristic_eval_c0 0 rule_heuristic

# ---------------------------------------------------------------- 4. Sensitivity (Fig. 6) and layout perturbation (Table 6)
for s in $SEEDS; do
  for k in 1 3 10 20; do python scripts/train_eval/s8_cv.py --run sens_k${k}_s$s --seed $s $MAIN --cache feat_cache_v3 --topk $k; done
  for f in 25 50 75; do python scripts/train_eval/s8_cv.py --run sens_frac${f}_s$s --seed $s $MAIN --cache feat_cache_v3 --train-frac 0.$f; done
  for g in 003:0.03 03:0.3 1:1.0; do
    python scripts/train_eval/s8_cv.py --run sens_mu${g%%:*}_s$s --seed $s --struct ${g##*:} --struct-epochs 20 --fixed-cost 0.0 --aug pair --cache feat_cache_v3
  done
  for n in 025:0.25 050:0.5 100:1.0; do
    python scripts/train_eval/build_perturbed.py noise${n%%:*}_s$s noise ${n##*:} $s           # also writes runs/episodic_noise*
    python scripts/train_eval/s8_cv.py --run layout_noise${n%%:*}_s$s --seed $s $MAIN --cache feat_cache_v3 --test-cache feat_cache_v3_noise${n%%:*}_s$s
  done
  python scripts/train_eval/build_perturbed.py randid_s$s randid 0 $s
  python scripts/train_eval/s8_cv.py --run layout_randid_s$s --seed $s $MAIN --cache feat_cache_v3 --test-cache feat_cache_v3_randid_s$s
done

# ---------------------------------------------------------------- 5. VLM baselines (vlm environment; Tables 1, 2 and 4)
M=$D/models
python scripts/baselines/run_direct.py --model $M/Qwen3-VL-8B-Instruct      --out $OUT/direct_8b
python scripts/baselines/run_direct.py --model $M/Qwen3-VL-32B-Instruct-FP8 --out $OUT/direct_32b
python scripts/baselines/run_direct.py --model $M/Qwen3-VL-8B-Instruct      --out $OUT/direct_8b_overlay  --overlay
python scripts/baselines/run_direct.py --model $M/Qwen3-VL-32B-Instruct-FP8 --out $OUT/direct_32b_overlay --overlay
python scripts/baselines/run_direct.py --model $M/InternVL3_5-8B-HF --family internvl --out $OUT/direct_internvl8b
for m in 8b:Qwen3-VL-8B-Instruct 32b:Qwen3-VL-32B-Instruct-FP8; do
  python scripts/baselines/run_direct.py --model $M/${m##*:} --out $OUT/direct_${m%%:*}_episodic --tasks episodic
done
python scripts/baselines/run_direct.py --model $M/InternVL3_5-8B-HF --family internvl --out $OUT/direct_internvl8b_episodic --tasks episodic
python scripts/baselines/run_structured.py --model $M/Qwen3-VL-32B-Instruct-FP8 --out $OUT/struct32b_text --no-frames
python scripts/baselines/run_structured.py --model $M/Qwen3-VL-32B-Instruct-FP8 --out $OUT/struct32b_frames
python scripts/baselines/run_structured.py --model $M/Qwen3-VL-32B-Instruct-FP8 --out $OUT/struct32b_text_v2 --no-frames --prompt v2

# ---------------------------------------------------------------- 6. Release files, tables and figures
python scripts/analysis/export_release.py               # writes $OUT/release_build; copy it to release/
python scripts/analysis/sens_summary.py release/figure_data/sens_summary.json
python scripts/analysis/layout_summary.py release/figure_data/layout_summary.json
python scripts/analysis/diag_oracle.py ens_final release/figure_data/diag_oracle.json
python scripts/analysis/check_release.py --ci           # Tables 1-4 from release/ only
(cd scripts/figures && python table_ci.py && python tables_extra.py && python fig_selective.py && python fig_errors.py && python fig_sensitivity.py)
