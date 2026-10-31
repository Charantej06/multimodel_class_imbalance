# AGENTS.md

## Global Rules
- Never synthesize, simulate, randomize or fake data or features. If a file is missing, corrupt or a Git LFS pointer, raise an error and STOP and tell me.
- Splits are by ACTOR. Scalers/PCA/GMM are fit on train only. Tune and select on validation Macro-F1 only; test is evaluated once at the val-selected checkpoint.
- No hardcoded class counts or modality names. Don't overwrite MELD outputs. Write CREMA-D outputs under cremad/.
- Scripts take --media_root (default CREMAD) and resolve files from the clip id: {media_root}/VideoFlash/{id}.flv and {media_root}/AudioWAV/{id}.wav. Don't store absolute paths.
- Do ONLY the step named below. When finished, summarize what you did, show evidence (command outputs), and STOP. Do not start the next step.

## Project Context
- **Project Goal**: Adapt the repo "multimodel_class_imbalance" (Fusion V2 -> 512-d latent -> RCS-Optimal -> MLP) to CREMA-D (6 emotions: anger, disgust, fear, happy, neutral, sad) using AUDIO + VIDEO (no text).
- **Backbones**: Pretrained and FROZEN; only fusion, RCS and the head are trained, on cached features.
- **Branch**: `cremad`.
- **Constraint**: MELD must keep working.

## Execution Plan
- **Step 0**: Setup + baseline
- **Step 1**: Audit
- **Step 2**: Data verification
- **Step 3**: Manifest + actor split
- **Step 4**: Generalize code (modalities, num_classes)
- **Step 5**: Video face-crop preprocessing
- **Step 6**: Audio features (WavLM, layer probe)
- **Step 7**: Video features + probes + feature sets + go/no-go gate
- **Step 8**: Train fusion + baselines + ablations
- **Step 9**: Latents + RCS + baselines (balanced and long-tailed)
- **Step 10**: Report

