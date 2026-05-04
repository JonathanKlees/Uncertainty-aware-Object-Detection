# Label Error Detection Benchmark

Benchmarks multiple **label error detection methods** by computing precision-recall curves against the label error ground truth (LEGT).

---

## Methods Compared

| Method | Description |
|--------|-------------|
| **Naive Baseline** | Flag all predictions below a score threshold as potential label errors |
| **Score Baseline** | Rank annotations by detection confidence score |
| **Loss Inspection** | Use training loss values to identify suspicious annotations |
| **ObjectLab** | Systematic label error detection using object-level analysis |

---

## Directory Structure

```
labelerrordetectionbenchmark/
└── precision_recall_curve/
    └── prec_rec_curve_all_methods.py   # Generate PR curves for all methods
```

---

## Configuration

The main script (`prec_rec_curve_all_methods.py`) defines:

- **LEGT variants** (ground truth for evaluation):
  - `less_restrictive` (Variant 1): prob ≥ 0.5
  - `stricter` (Variant 2): prob ≥ 0.8 AND height ≥ 40px

- **Detection model** (`PRED_NET`): Which model's predictions are used (e.g., `CascadeRcnn`, `GroundingDino`)

- **Label error proposals**: Per-method prediction files ranked by error confidence score

---

## Output

Precision-recall curves showing how well each method identifies true label errors at different confidence thresholds. Results are generated per dataset and per LEGT variant.
