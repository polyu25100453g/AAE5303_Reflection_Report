# AAE5303 Robust Control Technology in Low-Altitude Aerial Vehicle
## Post-Lesson Reflection Report

**Student Name:** [Your Name]  
**Student ID:** [Your Student ID]  
**Group Number:** [Your Group Number]  
**Date:** [Submission Date]

### Section 1: AI Usage Experience

In this project, I used Cursor mainly as a technical debugging partner for computer vision code, not as a writing tool. My most frequent use was in Module 3 segmentation, where the pipeline had multiple fragile points: class encoding consistency, loss behavior, mixed precision stability, and metric calculation logic.

I used AI for three high-value tasks. First, code-level diagnosis: when training loss decreased but validation mIoU plateaued, I asked Cursor to identify likely causes (class imbalance, ignore-label handling, augmentation-label mismatch, learning-rate split issues). Second, experiment scripting: I used it to quickly prepare controlled command variants (changing Dice loss weight, backbone/head LR ratio, FP16/TTA options). Third, metric traceability: I used it to verify that evaluation outputs actually matched what the training objective should improve.

I used Cursor almost daily during tuning weeks. The most useful features were structured chat reasoning and targeted code suggestions. Autocomplete helped, but the real value came from turning vague symptoms into concrete, testable checks.

### Section 2: Understanding AI Limitations

A clear limitation was metric interpretation drift. Our segmentation logs included `pixel_accuracy`, `mean_iou`, and `mean_dice`, sometimes with both "ALL classes" and "IGNORE background class 0" variants. AI initially gave a plausible explanation that treated these as roughly interchangeable indicators. That was misleading.

In practice:
- `pixel_accuracy` can be high even when minority classes are poor.
- `mean_iou` and `mean_dice` are sensitive to class frequency and background handling.
- background-ignored and all-class numbers are not directly comparable.

The answer sounded correct at first, but it underweighted statistical bias and protocol constraints. I detected this by checking confusion-matrix behavior and per-class IoU distributions, not just one global number. Several underrepresented classes were near-zero IoU while global pixel accuracy still looked good.

Another limitation appeared in hyperparameter suggestions. AI recommended "typical" values that are often reasonable but not always suitable for our dataset and compute budget. Some settings looked stable early but later produced overconfident background predictions and weak boundaries. I fixed this by treating AI output as hypothesis generation only; every suggestion required controlled A/B validation.

### Section 3: Engineering Validation

I validated AI-assisted changes through a strict experimental loop:

1. **Single-variable ablation**  
   I changed one factor at a time (Dice weight, LR split, crop policy, TTA) to avoid confounded conclusions.

2. **Per-class analysis before acceptance**  
   I accepted a run only if minority/edge-sensitive classes improved or at least did not collapse, not just because one global score increased.

3. **Numerical sanity checks**  
   I verified class index ranges, ignore-label masking, output scaling (0-1 vs 0-100), and confusion-matrix consistency.

4. **Failure reproduction**  
   For unstable runs, I reran with fixed seed and identical config to separate systematic errors from noise.

5. **Inference/training consistency**  
   I checked whether validation gains survived final inference settings (FP16/TTA). Some "improvements" did not survive deployment-style inference.

This process changed how I used AI: from "answer provider" to "fast hypothesis generator under strict verification."

### Section 4: Problem-Solving Process

The hardest technical issue was a performance ceiling with unstable class behavior: training loss dropped, but mIoU gains stalled and boundaries remained coarse for thin/small structures.

**Initial symptom**
- Loss curves looked healthy.
- Pixel accuracy was relatively high.
- Visual masks still had boundary bleeding.
- Minority classes were frequently absorbed into dominant classes.

**First failed attempts**
I first followed a common AI suggestion: train longer and increase augmentation strength. Result: marginal improvement on dominant classes, weak improvement on hard classes, and occasional label-edge artifacts.  
Then I increased Dice auxiliary weight aggressively. Result: short-term boundary gain but less stable convergence and calibration drift on some classes.

**Root-cause analysis**
I switched to systematic diagnosis:
1. verify GT label encoding and prediction class space alignment;
2. verify training masking vs evaluation ignore-label behavior;
3. inspect per-class IoU trajectories per epoch;
4. inspect qualitative masks for recurring error modes;
5. rebalance optimization between pretrained backbone and decoder head.

**What worked**
The most reliable improvements came from:
- lower backbone LR relative to head,
- moderate (not aggressive) Dice auxiliary weight,
- balanced augmentation that preserved label fidelity,
- resume/checkpoint discipline to avoid losing good states,
- per-class IoU gating during model selection.

The breakthrough was not one dramatic trick; it was reducing fragility through disciplined, class-aware iteration.

### Section 5: Learning Growth

My biggest growth is debugging model behavior beyond aggregate metrics. Earlier, I treated one score as ground truth. Now I require joint evidence: per-class metrics, qualitative mask inspection, and consistency between training objective and evaluation protocol.

I also improved at failure decomposition:
- optimization instability vs data/label mismatch,
- metric-definition mismatch vs true model weakness,
- training improvements vs inference-time robustness.

This decomposition made my experiments faster and more reliable. I now run fewer broad "try everything" experiments and more targeted ablations, which reduced wasted training cycles significantly.

### Section 6: Critical Reflection

AI improved my speed, but it also increased the risk of false confidence. It generates plausible code and parameter suggestions quickly, which is useful for exploration. But decisive progress came from manual verification, metric literacy, and experimental discipline.

Early in the semester, I sometimes over-trusted generic "best practices." Later, I changed my workflow: AI suggestions became hypotheses, not conclusions. That shift made AI genuinely valuable.

If I redo this project, I will:
1. define acceptance criteria (including per-class thresholds) before tuning,
2. enforce one-change-at-a-time ablations from the beginning,
3. require each AI recommendation to state its potential failure mode.

AI helped a lot, but only when paired with technical skepticism.

### Section 7: Evidence (Optional but Recommended)

#### 7.1 Metric Verification Code (Class-wise + Aggregate Consistency)

```python
import json
import numpy as np

# Example: read exported per-class IoU (None for absent classes)
with open("output/per_class_iou.json", "r", encoding="utf-8") as f:
    per_class = json.load(f)  # {"0": 0.91, "1": 0.72, ...}

ious = np.array([v for v in per_class.values() if v is not None], dtype=float)
miou_recomputed = float(np.mean(ious))

with open("output/segmentation_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

miou_reported = float(metrics["miou"])  # same scale as exported file
print("mIoU(recomputed) =", round(miou_recomputed, 4))
print("mIoU(reported)   =", round(miou_reported, 4))
print("Absolute diff    =", round(abs(miou_recomputed - miou_reported), 6))
```

This check catches scale/filter inconsistencies between per-class export and aggregate report.

#### 7.2 "Pixel Accuracy Trap" Guard

```python
import json

with open("output/segmentation_metrics.json", "r", encoding="utf-8") as f:
    m = json.load(f)

pixel_acc = m.get("pixel_accuracy", None)   # available in some logs
miou = m["miou"]
dice = m["dice_score"]

# Warning heuristic; thresholds are task-dependent
if pixel_acc is not None and pixel_acc > 0.90 and miou < 0.60:
    print("[WARN] Possible class-imbalance illusion: high pixel_acc but low mIoU.")
if dice < miou:
    print("[INFO] Dice lower than mIoU in this run; inspect boundary predictions.")
```

This prevents overestimating model quality when dominant classes inflate global accuracy.

#### 7.3 Leaderboard JSON Schema Validation

```python
import json

required_top = {"group_name", "project_private_repo_url", "metrics"}
required_metrics = {"miou", "dice_score", "fwiou"}

path = "module3_segmentation/leaderboard/submission_unet_Deepthinker.json"
with open(path, "r", encoding="utf-8") as f:
    sub = json.load(f)

missing_top = required_top - set(sub.keys())
assert not missing_top, f"Missing top-level keys: {missing_top}"

missing_metrics = required_metrics - set(sub["metrics"].keys())
assert not missing_metrics, f"Missing metric keys: {missing_metrics}"

for k in required_metrics:
    assert isinstance(sub["metrics"][k], (int, float)), f"{k} must be numeric"

assert sub["project_private_repo_url"].startswith("https://github.com/")
assert sub["project_private_repo_url"].endswith(".git")

print("Submission JSON schema check passed.")
```

This removes format-level failure risk before final submission.

#### 7.4 Controlled Ablation Launcher (Single-Variable Change)

```bash
#!/usr/bin/env bash
set -e

# Baseline
python3 scripts/train_deeplab_uavscenes.py \
  --backbone-lr-ratio 0.1 --dice-weight 0.2 --epochs 40 --exp-name base

# Ablation A: Dice weight only
python3 scripts/train_deeplab_uavscenes.py \
  --backbone-lr-ratio 0.1 --dice-weight 0.4 --epochs 40 --exp-name dice04

# Ablation B: LR split only
python3 scripts/train_deeplab_uavscenes.py \
  --backbone-lr-ratio 0.05 --dice-weight 0.2 --epochs 40 --exp-name lr005
```

This script enforces reproducible one-factor comparisons and reduces confounding.

#### 7.5 Prompt/Validation Example

- Prompt: "Given my segmentation logs, can I use pixel accuracy as a leaderboard metric substitute?"
- AI tendency: "Pixel accuracy is useful and correlated."
- Validation action: checked template keys, evaluation output definition, and per-class IoU.
- Final conclusion: pixel accuracy is diagnostic only; required fields are `miou`, `dice_score`, `fwiou`.