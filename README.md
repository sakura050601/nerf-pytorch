NeRF with TrueDepth RGB-D Datasets

This repository is a customized version of NeRF (Neural Radiance Fields) training code adapted for iOS TrueDepth-based RGB-D scans.

It provides:
	•	A dataset converter that transforms TrueDepth RGB-D + pose + intrinsics into a Blender-style NeRF dataset.
	•	Training configs for multiple TrueDepth sessions (e.g., trueDepth_1, trueDepth_2).
	•	Auto device selection (MPS on macOS, CUDA on Linux/Windows, CPU fallback).
	•	Optional background removal pipeline for preprocessing RGB images.

⸻

1. Environment Setup

This project assumes Python 3.9–3.11 and a working conda/mamba environment (recommended).

1.1 Create a dedicated environment

conda create -n nerf python=3.10
conda activate nerf

1.2 Install PyTorch

Follow the official PyTorch instructions if needed. Typical examples:

macOS (Apple Silicon / Intel)

# Example: CPU/MPS build (check PyTorch docs for the latest commands)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

If you want GPU acceleration on Apple Silicon, make sure your PyTorch build has MPS support enabled and that your macOS + Xcode + drivers meet the requirements.

Windows / Linux
For NVIDIA GPUs (CUDA):

# Example for CUDA 11.8 (adjust according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For CPU-only:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

1.3 Install Python dependencies

If there is a requirements.txt:

pip install -r requirements.txt

If you use the optional background removal step for RGB preprocessing, install:

# Generic CPU (Windows/Linux)
pip install rembg onnxruntime

# On macOS / Apple Silicon, you may need:
pip install rembg onnxruntime-silicon

# And in some cases keep NumPy < 2 for compatibility:
pip install "numpy<2"

If you see errors like AttributeError: _ARRAY_API not found or ImportError: numpy.core.multiarray failed to import, it often means the wheel was compiled against NumPy 1.x. In that case, downgrade to numpy<2 (or upgrade the affected package once it supports NumPy 2.x).

⸻

2. Directory Structure (Typical)

A typical structure looks like:

.
├─ run_nerf.py
├─ convert_dataset.py
├─ configs/
│  ├─ trueDepth1.txt
│  └─ trueDepth2.txt
├─ data/
│  ├─ trueDepth_1/        # converted NeRF-style dataset
│  └─ trueDepth_2/
└─ source_data/
   ├─ trueDepth_1/        # raw TrueDepth session (pose_*.txt, intrinsics_*.txt, color_*.png, depth_*.png)
   └─ trueDepth_2/

You can adapt the folder names as needed as long as your configs (configs/*.txt) point to the correct datadir.

⸻

3. Converting TrueDepth Datasets

Raw TrueDepth data (from your AR capture app) is assumed to have files like:
	•	pose_i.txt (px, py, pz, qx, qy, qz, qw)
	•	intrinsics_i.txt (fx, fy, cx, cy)
	•	color_i.png
	•	depth_i.png

Use convert_dataset.py to convert them into NeRF-style datasets.

3.1 Convert trueDepth_1

python convert_dataset.py \
  --input_dir source_data/trueDepth_1 \
  --output_dir data/trueDepth_1 \
  --train_ratio 0.34 --val_ratio 0.33 --test_ratio 0.33

This creates:
	•	data/trueDepth_1/train/…
	•	data/trueDepth_1/val/…
	•	data/trueDepth_1/test/…
	•	data/trueDepth_1/transforms_train.json
	•	data/trueDepth_1/transforms_val.json
	•	data/trueDepth_1/transforms_test.json

with:
	•	color images: r_<index>.png
	•	depth maps: r_<index>_depth_0001.png
	•	normal maps (estimated): r_<index>_normal_0001.png

3.2 Convert trueDepth_2

python convert_dataset.py \
  --input_dir source_data/trueDepth_2 \
  --output_dir data/trueDepth_2 \
  --train_ratio 1 --val_ratio 1 --test_ratio 1

This exposes the entire dataset to all splits (train/val/test). This is convenient for small experiments or for cases where you want flexible evaluation on the same frames.

⸻

4. Training NeRF

4.1 Train on trueDepth_1

python run_nerf.py --config configs/trueDepth1.txt

configs/trueDepth1.txt should contain at least:

expname = blender_paper_truedepth1
basedir = ./logs
datadir = ./data/trueDepth_1
dataset_type = blender

no_batching = True
use_viewdirs = True
white_bkgd = True
half_res = True

N_samples = 32
N_importance = 32
N_rand = 512

lrate = 1e-4
lrate_decay = 500

render_factor = 4
i_testset = 1000
i_print = 100
i_img = 500
i_weights = 10000

You can tune these hyperparameters as needed.

4.2 Train on trueDepth_2

python run_nerf.py --config configs/trueDepth2.txt

configs/trueDepth2.txt should be similar, but with:

expname   = blender_paper_truedepth2
datadir   = ./data/trueDepth_2
dataset_type = blender
...


⸻

5. Fine-tuning on a Related Dataset

If you already trained on trueDepth_1 and have a checkpoint such as:

./logs/blender_paper_truedepth1/200000.tar

you can fine-tune on trueDepth_2:

python run_nerf.py \
  --config configs/trueDepth2.txt \
  --expname blender_paper_truedepth2_finetune \
  --datadir ./data/trueDepth_2 \
  --dataset_type blender \
  --ft_path ./logs/blender_paper_truedepth1/200000.tar

	•	--ft_path tells the script to load that checkpoint’s weights.
	•	In your modified run_nerf.py, if ft_path is set, it loads weights only and resets optimizer / global_step (true fine-tune mode).

⸻

6. Long-running Training with nohup (Server / Remote)

On a remote machine or server, you can keep training running in the background:

nohup python -u run_nerf.py \
  --config configs/trueDepth1.txt \
  > train_nerf.log 2>&1 &

	•	-u: unbuffered output (flush logs immediately).
	•	> train_nerf.log 2>&1: redirect both stdout and stderr to train_nerf.log.
	•	Training continues even if you disconnect.

Monitor progress with:

tail -f train_nerf.log


⸻

7. Device Selection (Mac vs. Windows / Linux)

run_nerf.py automatically selects the device:
	•	macOS:
	•	If torch.backends.mps.is_available() → use mps.
	•	Otherwise, fall back to CPU.
	•	Other platforms:
	•	If torch.cuda.is_available() → use cuda.
	•	Otherwise, fall back to CPU.

You do not need to pass any --device flag: the script chooses the best available option at runtime and logs it (e.g., to wandb).

⸻

8. Optional: Background Removal Preprocessing

If your TrueDepth RGB images contain complex backgrounds and you want to keep only the main object, you can run a preprocessing script using rembg + onnxruntime before conversion.

Example (single image):

from rembg import remove
from PIL import Image

input_path = 'data/trueDepth_2/test/r_290.png'
output_path = 'data/trueDepth_2/test/r_290_fg.png'

with Image.open(input_path) as img:
    out = remove(img)
    out.save(output_path)

You can extend this into a batch script that:
	1.	Reads all r_*.png in a folder.
	2.	Applies remove().
	3.	Saves foreground-only images (e.g., with _fg suffix).
	4.	Uses these cleaned images as input for convert_dataset.py (or swaps the original color_i.png with the cleaned version before conversion).

Note: On Apple Silicon, use onnxruntime-silicon and a NumPy version compatible with the wheel used by onnxruntime.

⸻

9. Logging & Visualization

The training loop is integrated with Weights & Biases (wandb):
	•	Each training run logs:
	•	Loss, PSNR, learning rate.
	•	Optionally a rendered validation view (val_rgb) every i_img steps.

You can configure your W&B project name inside run_nerf.py:

wandb.init(
    project="nerf_truedepth",
    name=args.expname,
    config=vars(args),
)

Make sure you have logged in to W&B:

pip install wandb
wandb login


⸻

10. Notes & Tips
	•	For small datasets like trueDepth_2, consider:
	•	Smaller N_rand and shorter total steps.
	•	Higher train_ratio (even 1.0) if you mainly care about reconstruction quality, not generalization.
	•	For high-quality reconstructions:
	•	Ensure camera paths have enough parallax.
	•	Use consistent lighting.
	•	Avoid extreme motion blur when capturing RGB-D frames.
	•	If training is unstable:
	•	Lower learning rate (lrate).
	•	Increase number of iterations gradually.
	•	Check your near/far bounds and scene scale.

⸻

Citation
Kudos to the authors for their amazing results:

@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
However, if you find this implementation or pre-trained models helpful, please consider to cite:

@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}