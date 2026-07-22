# Step 5 — Headless finished-print inspection worker

This step adds a file-based worker around the one-shot zero-shot inspector.
It does not connect to the printer, Tulip, robots, SQLite, or the React UI yet.

## Contract

The worker watches one jobs directory. Every inspection has its own folder:

```text
runs/_inspection/jobs/<inspection_id>/
├── request.json
└── source.jpg
```

The request format is:

```json
{
  "schema_version": 1,
  "inspection_id": "manual-job-001",
  "source_image": "source.jpg",
  "expected_part": "test-part",
  "prompt": "3D printed part . plastic object",
  "quality_threshold": 750,
  "identity_threshold": 1000
}
```

The two threshold fields are optional. When omitted, the worker uses the
values from `src/config.py`.

After processing, the same folder contains:

```text
runs/_inspection/jobs/<inspection_id>/
├── request.json
├── source.jpg
├── mask.png
├── annotated.jpg
└── result.json
```

A completed `result.json` prevents the same job from running twice.

## Manual one-job test

```powershell
New-Item -ItemType Directory -Force `
  .\runs\_inspection\jobs\manual-job-001

Copy-Item `
  .\test_images\finished-part.jpg `
  .\runs\_inspection\jobs\manual-job-001\source.jpg

@'
{
  "schema_version": 1,
  "inspection_id": "manual-job-001",
  "source_image": "source.jpg",
  "expected_part": "test-part",
  "prompt": "3D printed part . plastic object"
}
'@ | Set-Content `
  .\runs\_inspection\jobs\manual-job-001\request.json `
  -Encoding utf8
```

Process all currently pending jobs and then exit:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\src\inspection_worker.py `
  --jobs-dir ".\runs\_inspection\jobs" `
  --references ".\references" `
  --once
```

Open the result:

```powershell
Get-Content `
  .\runs\_inspection\jobs\manual-job-001\result.json

Start-Process `
  .\runs\_inspection\jobs\manual-job-001\annotated.jpg
```

## Long-running mode

Omit `--once` to keep the models and FAISS references loaded while the worker
waits for new jobs:

```powershell
python .\src\inspection_worker.py `
  --jobs-dir ".\runs\_inspection\jobs" `
  --references ".\references"
```

Stop it with `Ctrl+C`.
