# IAE-WatchAct

This repository contains the implementation and public evaluation artifacts for *Instance-anchored interaction evidence for behavior-grounded robot planning*. IAE registers object instances in the final scene, propagates their identities through the demonstration, extracts hand and body interaction evidence, and decodes a constrained placement program. The method is trained with activity-level outcomes and does not require frame annotations.

The accompanying paper is the source of the reported protocol and should be cited together with this repository.

## Installation

Use Python 3.11 or newer. The light-weight package can be installed without the private dataset:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

For a full reproduction, install the environment matching the stage being run:

- `requirements-main.txt`: NumPy, PyTorch and shared numerical dependencies.
- `requirements-perception.txt`: Grounding DINO tiny, SAM 2.1 hiera-tiny, MediaPipe Hands and RTMPose-m. Install each project from its official release and keep its weights outside this repository.
- `requirements-vlm.txt`: Transformers and the VLM runtime for Qwen3-VL-8B/32B (including the FP8 32B setup) and InternVL3.5-8B.
- A separate WatchAct evaluation environment: clone `Baiqi-Li/WatchAct` at commit `7036927a94a160e7d420e3f7156cd5b2e9e8f3e6` and point `IAE_WATCHACT_ROOT` at the checkout.

Set paths through environment variables or a local copy of `configs/paths.example.yaml`:

```bash
export IAE_DATA_ROOT=/path/to/watchact-data
export IAE_OUT_ROOT=/path/to/watchact-data/v2
export IAE_WATCHACT_ROOT=/path/to/WatchAct
export IAE_MODEL_ROOT=/path/to/models
```

No video, annotation, or model weight is included. Obtain WatchAct and the model weights from their official sources and comply with their licenses.

## Reproduction order

Run the stages from the repository root after installing the relevant environment:

```bash
python scripts/pipeline/s1_frames.py --help
python scripts/pipeline/s2_detect_final.py --help
python scripts/pipeline/s3_register.py --help
python scripts/pipeline/s4_track.py --help
python scripts/pipeline/s5a_person.py --help
python scripts/pipeline/s5b_hands.py --help
python scripts/pipeline/s5c_body.py --help
python scripts/train_eval/build_cache.py --help
python scripts/train_eval/s8_cv.py --run iae_cv --folds random
python scripts/train_eval/ensemble.py --help
python scripts/baselines/run_direct.py --model /path/to/model --out /path/to/output --help
python scripts/analysis/dump_pairs.py /path/to/pairs.json
python scripts/analysis/test_dp_bruteforce.py
```

The supplied `release/folds.json` records the random seed-0 activity split and the recording-order blocked split. The supplied prediction files contain only request identifiers, task/view metadata, canonical predicted pairs, and score flags. They can be inspected without the private data.

## Paper cross-reference

| Paper item | Script or artifact | Output |
|---|---|---|
| Main IAE and direct baselines | `scripts/analysis/dump_pairs.py` | `release/predictions/pairs_all.json` |
| IAE ensemble rows | `scripts/train_eval/ensemble.py` | `release/predictions/iae_ensemble_rows.json` |
| Table 1 direct methods | `scripts/baselines/score_direct.py` | `release/predictions/direct_rows.json` |
| Activity folds | `scripts/train_eval/s8_cv.py` | `release/folds.json` |
| Dynamic-programming check | `scripts/analysis/test_dp_bruteforce.py` | test output |
| Figures and tables | `scripts/figures/` | `figs/` |

## Reported reference numbers

The public rows reproduce the paper's aggregate flags on 455 implicit-intent requests:

| Method | NC success / strict | RD success / strict | Overall success / strict |
|---|---:|---:|---:|
| IAE ensemble | 42.6 / 16.9 | 80.4 / 74.2 | 64.2 / 49.7 |
| Qwen3-VL-32B direct | 22.6 / 4.6 | 31.2 / 23.5 | 27.5 / 15.4 |
| 32B + evidence | 35.9 / 12.3 | 73.5 / 68.5 | 57.4 / 44.4 |

## Hardware and limitations

Perception requires a CUDA GPU and the four model families above. The FP8 VLM path is intended for a GPU with sufficient device memory; CPU execution is supported only for small unit tests and bookkeeping. End-to-end timing depends on video decoding, detector/segmenter versions, and GPU memory. The released scores are tied to the pinned WatchAct simulator, its public scene conventions, and the listed model revisions. Episodic-task prediction rows are not included in this release.

## License

The released code is MIT licensed. Third-party components retain their upstream licenses.


Run the unit test with `python -m pytest tests`.
