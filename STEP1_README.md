# Step 1 — Headless one-image zero-shot inspection

## Exactly what this changes

This step adds a new **offline/headless inspection path** to
`ICHR-BME/zero-shot-robotic-pipeline`. It does not change or remove the existing
GUI, robots, side camera, homography, or SQLite code.

New files:

- `src/inspection_contract.py`: pure APPROVED/REJECTED/UNKNOWN decision rules.
- `src/inspection.py`: Grounded-SAM → masked ROI → DINOv2 → FAISS pipeline for
  one image.
- `src/inspect_image.py`: command-line smoke test that writes JSON and images.
- `tests/test_inspection_contract.py`: tests the decision boundaries without AI
  models or hardware.

This is intentionally the first milestone because we need to prove this stable
contract before touching Bambu `FINISH`, UCDavisGUI, or Tulip:

```text
finished_part.jpg + reference folders -> result.json
```

## Copy the files

Copy the contents of this patch into the root of your local
`zero-shot-robotic-pipeline` clone, preserving the `src/` and `tests/` paths.
No existing file is overwritten in Step 1.

## Reference-folder layout

```text
references/
├── gear-v3/
│   ├── reference-01.jpg
│   └── reference-02.jpg
└── bracket-v2/
    ├── reference-01.jpg
    └── reference-02.jpg
```

The folder name is the `part_id` returned by the inspector.

## Run it

From the repository root:

```powershell
python src/inspect_image.py `
  --image test_images/finished_gear.jpg `
  --references references `
  --expected-part gear-v3 `
  --quality-threshold 40 `
  --identity-threshold 120 `
  --output runs/manual-inspection
```

Bash equivalent:

```bash
python src/inspect_image.py \
  --image test_images/finished_gear.jpg \
  --references references \
  --expected-part gear-v3 \
  --quality-threshold 40 \
  --identity-threshold 120 \
  --output runs/manual-inspection
```

Expected output:

```text
runs/manual-inspection/
├── source.jpg
├── mask.png          # absent only when segmentation failed
├── annotated.jpg
└── result.json
```

## Decision semantics

Lower FAISS L2 distance is better.

```text
distance > identity_threshold                 -> UNKNOWN
recognized part != expected part              -> REJECTED / WRONG_PART
distance > quality_threshold                  -> REJECTED
otherwise                                     -> APPROVED
```

The configuration is invalid when `quality_threshold > identity_threshold`.

## Run the pure tests

Install pytest if it is not already installed:

```bash
pip install pytest
```

Because `src/` is not yet an installed package, expose it on `PYTHONPATH`:

PowerShell:

```powershell
$env:PYTHONPATH="src"
pytest -q tests/test_inspection_contract.py
```

Bash:

```bash
PYTHONPATH=src pytest -q tests/test_inspection_contract.py
```

## What Step 1 does not do

- It does not listen for `gcode_state == FINISH`.
- It does not open a camera automatically.
- It does not write to Tulip.
- It does not persist precomputed FAISS embeddings yet; references are embedded
  when the command starts.
- It does not alter the existing CustomTkinter/robot workflow.

After this command reliably produces correct `result.json` files, Step 2 is to
make the reference catalogue persistent and turn the same inspector into a
long-running worker. Only after that do we connect UCDavisGUI's FINISH event.
