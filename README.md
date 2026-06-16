# Two-Stage CSLR Training Repository

This codebase was developed in the context of my bachelor's thesis, as a side component focused on continuous sign language recognition and translation results.

This repository contains the training and evaluation code for a two-stage Catalan continuous sign language recognition pipeline:

1. `video_to_glosses` trains the first-stage CTC model that maps sign video to glosses.
2. `glosses_to_text` trains the second-stage sequence-to-sequence model that maps glosses to Catalan text.
3. `cascade_glosses` evaluates the end-to-end cascade by combining both stages.

The repository has been cleaned so it only keeps source code and launch scripts. Generated artifacts such as checkpoints, logs, `wandb` runs, and notebook checkpoints are intentionally ignored by `.gitignore`.

## Repository Layout

```text
.
├── glosses_to_text/
│   ├── train_gloss_to_text.py
│   ├── train_gloss_to_text_baseline.py
│   └── train_model.sh
├── video_to_glosses/
│   ├── train_model.sh
│   ├── evaluate_model.sh
│   └── video_to_glosses/
│       ├── train.py
│       ├── evaluate.py
│       ├── dataset.py
│       ├── model.py
│       ├── metrics.py
│       └── video_augmentations.py
└── cascade_glosses/
    ├── cascade_evaluate.py
    └── evaluate.sh
```

## Setup Notes

The shell scripts now use placeholder paths and environment variables instead of machine-specific absolute paths. Before running a script, set the paths for your environment and data.

Typical variables used by the scripts:

- `VENV_ACTIVATE`: path to your Python environment activation script
- `G2T_DATA_DIR`: gloss-to-text dataset directory
- `G2T_OUTPUT_DIR`: gloss-to-text output directory
- `CSLR_INPUT_DIR`: video-to-glosses dataset directory
- `CSLR_OUTPUT_DIR`: video-to-glosses output directory
- `CHECKPOINT_PATH`: video-to-glosses checkpoint for evaluation
- `DATA_FILE`: JSON split file for evaluation
- `V2G_CHECKPOINT`: trained video-to-glosses checkpoint
- `G2T_MODEL_DIR`: trained gloss-to-text model directory
- `VOCAB_PATH`: vocabulary JSON used by the cascade

The scripts include placeholder defaults such as `/path/to/...` so it is clear what should be replaced.

## Expected Data Formats

### Stage 1: Video to Glosses

The `video_to_glosses` code expects JSON files for `train.json`, `val.json`, and `test.json` with one item per sample.

Required fields per sample:

- `video_path`: path to the video file
- `gloss_sequence`: list of gloss tokens in order
- `timeline`: list of frame-level alignment entries with `gloss`, `start`, and `end`

Example:

```json
{
  "video_path": "/data/videos/sample_001.mp4",
  "gloss_sequence": ["BON-DIA", "COM", "ESTAS"],
  "timeline": [
    {"gloss": "BON-DIA", "start": 0.0, "end": 1.1},
    {"gloss": "COM", "start": 1.1, "end": 1.5},
    {"gloss": "ESTAS", "start": 1.5, "end": 2.0}
  ]
}
```

The dataset loader uses:

- `video_path` to load the video
- `gloss_sequence` to build the CTC target sequence
- `timeline` to build frame-level supervision

The folder also expects a `vocab.json` file that maps tokens to integer ids.

### Stage 2: Glosses to Text

The `glosses_to_text` scripts expect TSV files named `train.tsv` and `validation.tsv`.

Required columns:

- `gloss_input`: input gloss sequence as text
- `output`: target Catalan sentence

Example TSV rows:

```tsv
gloss_input	output
BON-DIA COM ESTAS	Bon dia, com estàs?
ANAR CASA	Vaig a casa.
```

The gloss sequences are read as plain text and the model is trained to generate the `output` sentence.

### Stage 3: Cascade Evaluation

The cascade expects the same JSON format used by stage 1, plus a translation field.

Required fields per sample:

- `video_path`
- `gloss_sequence`
- `timeline`
- `translation`: reference Catalan translation used for evaluation

Example:

```json
{
  "video_path": "/data/videos/sample_001.mp4",
  "gloss_sequence": ["BON-DIA", "COM", "ESTAS"],
  "timeline": [
    {"gloss": "BON-DIA", "start": 0.0, "end": 1.1},
    {"gloss": "COM", "start": 1.1, "end": 1.5},
    {"gloss": "ESTAS", "start": 1.5, "end": 2.0}
  ],
  "translation": "Bon dia, com estàs?"
}
```

The cascade also needs a `vocab.json` file for the gloss vocabulary and a trained `final_model` directory from the gloss-to-text stage.

## Running the Stages

Typical usage:

- `glosses_to_text/train_model.sh` trains the gloss-to-text model.
- `video_to_glosses/train_model.sh` trains the video-to-gloss model.
- `video_to_glosses/evaluate_model.sh` evaluates a trained video-to-gloss checkpoint.
- `cascade_glosses/evaluate.sh` runs the full cascade evaluation.

Before submitting a job, replace the placeholder paths or export the matching environment variables.

## Notes

- Generated folders such as `output/`, `final_model/`, `logs/`, `train_logs/`, and `wandb/` should not be committed.
- The placeholder paths are intentional so new users can see exactly what needs to be configured.
- If you want this repository to be fully self-contained, the next useful step is to add a small `config/` or `data/` example directory with empty templates for the expected JSON and TSV files.

## GitHub Pages Report

The repository includes a standalone `report.html` file with the results analysis for the medical LSC translation work. To publish it on GitHub Pages together with the codebase:

1. Push the repository to GitHub.
2. In the repository settings, enable GitHub Pages from the branch you want to publish.
3. Use the repository root as the Pages source so both the codebase and `report.html` are available.

The root `index.html` file redirects visitors to `report.html`, so the published Pages site opens directly on the report while still keeping the rest of the repository accessible.
