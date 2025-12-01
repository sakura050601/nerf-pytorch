Got it～ here’s the pure English version you can drop into your README:

## Usage / Training Tips

### 1. Convert TrueDepth datasets

First, convert the raw TrueDepth data into the format expected by NeRF:

```bash
python convert_dataset.py \
  --input_dir source_data/trueDepth_1 \
  --output_dir data/trueDepth_1 \
  --train_ratio 0.34 --val_ratio 0.33 --test_ratio 0.33

The command above performs a standard train/val/test split for trueDepth_1.

python convert_dataset.py \
  --input_dir source_data/trueDepth_2 \
  --output_dir data/trueDepth_2 \
  --train_ratio 1 --val_ratio 1 --test_ratio 1

The command above exposes the full trueDepth_2 dataset to train/val/test simultaneously (useful for small experimental datasets).

⸻

2. Train NeRF on trueDepth_1

python run_nerf.py --config configs/trueDepth1.txt

configs/trueDepth1.txt should specify the data path (e.g., datadir=data/trueDepth_1), network architecture, and training hyperparameters.

⸻

3. Train NeRF on trueDepth_2

python run_nerf.py --config configs/trueDepth2.txt

Similarly, configs/trueDepth2.txt should be configured for data/trueDepth_2.

⸻

4. Long-running training with nohup (server / remote)

On a server or remote environment, you can run training in the background and log output to a file:

nohup python -u run_nerf.py \
  --config configs/trueDepth1.txt \
  > train_nerf.log 2>&1 &

	•	-u: unbuffered output so logs are written in real time.
	•	> train_nerf.log 2>&1: redirect both stdout and stderr to the same log file.
	•	The process will keep running after you close the terminal. You can monitor training with:

tail -f train_nerf.log



⸻

5. Device selection (Mac vs. other platforms)

The training script automatically selects the compute device:
	•	On macOS with Apple Silicon and Metal available, it prefers MPS.
	•	On Linux/Windows with a supported GPU, it prefers CUDA.
	•	If neither MPS nor CUDA is available, it falls back to CPU.

You do not need to manually pass a --device flag; the script will choose the best available option at runtime.

