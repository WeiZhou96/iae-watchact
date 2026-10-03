# IAE: instance-anchored interaction evidence for behavior-grounded robot planning

Code, cross-validation folds and per-request results for the paper *Instance-anchored interaction evidence for
behavior-grounded robot planning* (X. Xiao, B. Yang, L. Yang, W. Zhou), evaluated on the
[WatchAct](https://github.com/Baiqi-Li/WatchAct) benchmark.

A robot often has to act on what a person has shown rather than said: which of several identical boxes was pointed at,
or which one was handled. The plan is executed in the final scene of the demonstration, while the evidence occurs earlier,
possibly on objects that have moved since. IAE registers every object of the final scene to its public identifier, keeps
each identity through the video by backward mask propagation, and describes every frame by the geometry between the hands,
the forearms and these instances. An evidence network trained only from task outcomes scores the instances; a
grammar-constrained dynamic program decodes object–destination programs for pointing tasks, and symbolic programs handle
reference disambiguation and, without learning, the episodic tasks.

| Implicit-intent tasks (455 requests) | Plan success (%) | Strict success (%) |
|---|---:|---:|
| Qwen3-VL-32B, direct planning | 27.5 | 15.4 |
| Qwen3-VL-32B given the IAE evidence as explained text | 57.4 | 44.4 |
| **IAE**, three-seed ensemble | **64.2** | **49.7** |

On restoration, reversal and imitation (800 requests), IAE reaches 46.4% plan success without task-specific training,
against 27.0% for Qwen3-VL-32B.

## Contents

| Path | Content |
|---|---|
| `iae/` | Method: registration, features, evidence network, grammar-constrained decoding, RD and episodic programs, scoring |
| `iae/third_party/teb/` | Minimal subset of the TEB helper package (MIT): plan compiler, schema, request import, local VLM backend |
| `scripts/pipeline/` | Perception: request import, frames, detection, registration, mask propagation, person, hands, body |
| `scripts/train_eval/` | Feature caches, cross-validation training, ensembling, episodic programs, controls |
| `scripts/baselines/` | Direct and structured-evidence VLM baselines, hand-crafted baseline, scoring |
| `scripts/analysis/` | Statistics, diagnostics, timing, export of `release/`, and `check_release.py` |
| `scripts/figures/` | Scripts of the result tables and figures; `export_demo_cases.py` and `render_demo.py` render demonstration clips from the pipeline outputs and the WatchAct videos (requires ffmpeg) |
| `scripts/reproduce_paper.sh` | Every command behind the reported results, in order |
| `release/` | Request list, folds, and per-request predictions and scores of every reported method |
| `tests/` | Exhaustive check of the dynamic program; check of the released numbers |

## Check the reported numbers without data

For every request and method, the released files contain the predicted object–destination pairs and the WatchAct score.
The tables of the paper can be recomputed from them in a few seconds, without videos, models or the simulator:

```bash
pip install -e .
python scripts/analysis/check_release.py --ci     # Tables 1-4, including the bootstrap intervals
python -m pytest tests                             # dynamic program and release checks
```

## Installation

Python 3.11. The package itself needs only NumPy and Pillow; the stages of the pipeline use four environments, each with
this package installed (`pip install -e .`):

| Environment | Requirements | Used by |
|---|---|---|
| main | `requirements-main.txt` (PyTorch 2.11 with CUDA 12.8, Transformers 5.17) | detection, mask propagation, person boxes, training, programs, scoring |
| hands | `requirements-hands.txt` (MediaPipe 0.10.35) | `s5b_hands.py` |
| pose | `requirements-pose.txt` (rtmlib 0.0.16, ONNX Runtime 1.30) | `s5c_body.py` |
| vlm | `requirements-vlm.txt` (main environment with FP8 kernels) | VLM baselines |

The figure scripts additionally need Matplotlib (`pip install -e ".[figures]"`).

## Data, models and paths

Paths are read from the environment variables `IAE_DATA_ROOT`, `IAE_OUT_ROOT` and `IAE_WATCHACT_ROOT`, or from
`configs/paths.yaml` (a copy of `configs/paths.example.yaml`). The expected layout is:

```
$IAE_DATA_ROOT/
  data_full/WatchAct/                     WatchAct release, https://huggingface.co/datasets/BaiqiL/WatchAct (data/, meta_data/, videos/)
  manifests/                              written by scripts/pipeline/s0_import_watchact.py
  models/grounding-dino-tiny/             https://huggingface.co/IDEA-Research/grounding-dino-tiny
  models/sam2-hiera-tiny/                 https://huggingface.co/facebook/sam2.1-hiera-tiny (Transformers format)
  models/mediapipe/hand_landmarker.task   https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
  models/Qwen3-VL-8B-Instruct/            https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
  models/Qwen3-VL-32B-Instruct-FP8/       https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8
  models/InternVL3_5-8B-HF/               https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
$IAE_WATCHACT_ROOT/                       https://github.com/Baiqi-Li/WatchAct at commit 7036927a94a160e7d420e3f7156cd5b2e9e8f3e6
$IAE_OUT_ROOT/                            pipeline outputs (frames10/, tracks/, hands/, feat_cache_*/, runs/, ...)
```

RTMPose-m is downloaded by rtmlib from OpenMMLab on first use. No video, annotation or model weight is distributed with
this repository; please obtain them from their sources and follow their licenses. The WatchAct simulator and scorer are
used from the official repository and are needed only to score new predictions.

Every request is identified by a 24-character ID derived from its task, WatchAct example ID, activity, instruction and
video (`s0_import_watchact.py`). `release/requests.json` maps these IDs to the WatchAct example IDs and videos.

## Reproducing the paper

`scripts/reproduce_paper.sh` lists every command, with the run names that the analysis scripts read:

1. **Perception** for all 1,255 requests: frames at 10 Hz, count-constrained Grounding DINO detection on the anchor frame,
   registration to the public layout, SAM 2.1 propagation from the anchor frame, person boxes, MediaPipe hand and
   RTMPose body keypoints, and the 28-dimensional feature cache.
2. **IAE**: five-fold activity-level cross-validation of the evidence network for three seeds
   (`s8_cv.py --struct 0.1 --struct-epochs 20 --fixed-cost 0.0 --aug pair --cache feat_cache_v3`), the three-seed
   ensemble, and the episodic programs.
3. **Ablations and controls** (Table 3), **sensitivity** (Fig. 6) and **layout perturbation** (Table 6).
4. **VLM baselines**: direct planning with Qwen3-VL-8B, Qwen3-VL-32B (FP8) and InternVL3.5-8B, with and without
   overlays, and the 32B model given the IAE evidence as text.
5. **Release files, tables and figures**: `export_release.py`, `check_release.py` and `scripts/figures/`.

Training is seeded, but GPU kernels are not bitwise deterministic, so retrained models can differ slightly from the
released results; the released predictions are the ones reported in the paper.

| Paper | Scripts | Released output |
|---|---|---|
| Tables 1 and 2 | `s8_cv.py`, `ensemble.py`, `scripts/baselines/`; numbers by `check_release.py`, `table_ci.py`, `tables_extra.py` | `predictions/pairs_all.json`, `figure_data/table_ci.json`, `figure_data/tables_extra.json` |
| Table 3 | section 3 of `reproduce_paper.sh` | `predictions/ablations.json` |
| Table 4 | `s11_episodic.py`, `run_direct.py --tasks episodic`, `score_episodic.py` | `predictions/episodic.json` |
| Table 5 (camera and reference) | `scripts/figures/tables_extra.py` | `figure_data/tables_extra.json` |
| Table 6 (layout perturbation) | `build_perturbed.py`, `layout_summary.py` | `predictions/sensitivity.json`, `figure_data/layout_summary.json` |
| Table 7 (computation) | `scripts/analysis/timing.py`, `timing_cpu.py` | `figure_data/timing20.json` |
| Figs. 2 and 3 (examples) | `export_cases.py`, `export_qual2.py`, `fig_cases.py`, `fig_qual2.py` | need the WatchAct frames |
| Fig. 4 (reliability) | `scripts/figures/fig_selective.py` | `predictions/iae_ensemble_rows.json`, `predictions/direct_rows.json` |
| Fig. 5 (errors) | `scripts/figures/fig_errors.py` | `predictions/pairs_all.json` |
| Fig. 6 (sensitivity) | `sens_summary.py`, `scripts/figures/fig_sensitivity.py` | `figure_data/sens_summary.json` |
| Exactness of the dynamic program | `tests/test_dp_bruteforce.py` | comparison with exhaustive search |

## Released files

| File | Content |
|---|---|
| `release/requests.json` | All 1,255 requests: ID, task, WatchAct example ID, activity, camera, spatial reference, video, goal pairs |
| `release/folds.json` | Activity-to-fold assignment: random five folds (seed 0) and recording-order blocks |
| `release/predictions/pairs_all.json` | NC/RD: predicted pairs, success, strict success and validity for every method of Table 1 (`iae_ens`, `iae_s0`–`iae_s2`, `handcrafted`, `d8`, `d32`, `d8o`, `d32o`, `dvl`, `s32t`, `s32f`, `s32t2`) |
| `release/predictions/ablations.json` | NC/RD: the same for every variant and seed of Table 3; `runs` maps each row to its run name |
| `release/predictions/sensitivity.json` | NC/RD: sensitivity runs (`sens_*`), layout perturbation (`layout_*`) and nested choice of the decoding cost (`nested_lambda_*`) |
| `release/predictions/episodic.json` | Episodic tasks: IAE, Qwen3-VL-8B/32B and InternVL3.5-8B, and IAE under layout perturbation |
| `release/predictions/iae_ensemble_rows.json` | Ensemble output with its confidence (program or selection margin) |
| `release/predictions/direct_rows.json` | Per-request scores of the direct baselines |
| `release/figure_data/` | Summaries read by the figure scripts |

Destinations use the WatchAct identifiers (`basket_1_contain_region`, `main_back_left_region`, `wooden_cabinet_1_top_region`, ...).

## Computation

On one RTX 5880 Ada GPU, with keypoints on the CPU, perception and planning take about 70 s for a 34-second video
(2.1 s per second of video), dominated by per-frame person detection and mask propagation; perception is shared by all
requests on a video. The evidence network and the programs take less than half a second per video.

## Limitations

The results are tied to the WatchAct simulator and its symbolic execution, to the public scene layouts used for
registration, and to the listed model versions. The method has not been evaluated in other scenes or on a physical robot.

## Citation

```bibtex
@article{xiao2026iae,
  title   = {Instance-anchored interaction evidence for behavior-grounded robot planning},
  author  = {Xiao, Xinliang and Yang, Bowen and Yang, Li and Zhou, Wei},
  journal = {Robotics and Autonomous Systems},
  note    = {Submitted},
  year    = {2026}
}
```

Please also cite WatchAct (Li et al., 2026, arXiv:2606.26443) when using the benchmark.

## License

The code is released under the MIT License (`LICENSE`). `iae/third_party/teb/` keeps its MIT notice
(`THIRD_PARTY_NOTICES.md`). WatchAct, its videos and the model weights are subject to their own terms.
