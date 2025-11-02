# Sign_language_Detection

A set of notebooks for sign-language classification using three approaches: ViViT (video transformer), TimeSformer (space–time transformer), and LSTM-based sequence models (including an attention variant).

This repository is intended as a starting point for training and evaluating sign-language recognition models on your own video dataset. The notebooks include data loading, preprocessing, model adaptation, training loops, and evaluation examples.

Notebooks
---------

- `ViVit_Fine_Tuning.ipynb`
	- Purpose: Fine-tune a ViViT-like video transformer (using a VideoMAE/ViViT-style model) for video-level sign classification.
	- Inputs: video files organized by class (or a prepared dataset folder). The notebook samples a fixed number of frames per clip (e.g., 8–32) and uses a feature extractor.
	- Outputs: trained PyTorch model checkpoint, validation metrics, and prediction utilities.

- `TimeSformer_Fine_Tuning.ipynb`
	- Purpose: Fine-tune a TimeSformer-style transformer that separates temporal and spatial attention for video classification.
	- Inputs/Outputs: similar to the ViViT notebook; focuses on a TimeSformer model implementation and training loop.

- `lstm and lstm with attention.ipynb`
	- Purpose: Extract keypoint/feature sequences (the notebook uses MediaPipe to extract landmarks), build sequences, and train LSTM models: a baseline LSTM and an LSTM with a custom attention layer.
	- Inputs: pre-extracted per-frame features or MediaPipe keypoint `.npy` files organized per video.
	- Outputs: trained Keras models (`.h5`), evaluation metrics, and plotting utilities.

Quick prerequisites
-------------------

- Python 3.8+ (recommended)
- For transformer notebooks: PyTorch, torchvision, `transformers`, `timm`, and `decord` (or `opencv`-based loader). These notebooks assume a GPU for reasonable training speed.
- For LSTM notebook: TensorFlow 2.x and MediaPipe (keypoint extraction). CPU is fine for small experiments; GPU helps for faster training.
- Common utilities: numpy, pandas, scikit-learn, opencv-python, matplotlib

Minimal install example (adjust versions as needed):

```bash
pip install torch torchvision transformers timm decord numpy pandas scikit-learn opencv-python matplotlib
pip install tensorflow==2.13.0 mediapipe  # for the LSTM notebook
```

Dataset layout
--------------

The notebooks expect a simple dataset layout: videos grouped by class. Example:

```
dataset/
	train/
		Hello/
			v_001.mp4
			v_002.mp4
		Thanks/
			v_003.mp4
	val/
		Hello/
		Thanks/
```

For the LSTM notebook, the pipeline extracts keypoints per frame and saves them as `.npy` files inside folders for each video; then sequences of fixed length (e.g., 30 frames) are constructed and used to train the LSTM.

Quick start
-----------

1. Open the folder and start Jupyter Lab / Notebook:

```bash
cd Sign_language_Detection
jupyter lab
```

2. Open the notebook you want to run (`ViVit_Fine_Tuning.ipynb`, `TimeSformer_Fine_Tuning.ipynb`, or `lstm and lstm with attention.ipynb`).

3. Edit dataset paths and parameters in the first cells (dataset root, number of frames, batch size, device settings).

4. Run cells sequentially. For transformer notebooks, verify you have GPU support (CUDA) before full training.

Notes and tips
--------------

- Start with a small subset of your dataset to verify the pipeline and avoid long debugging runs.
- Use pre-trained checkpoints for ViViT/TimeSformer and fine-tune with a low learning rate.
- For LSTM training: ensure sequences have consistent length (use padding or sliding windows) and normalize features.
- Use logging (TensorBoard or simple CSV logging) and save checkpoints frequently.

Model & training expectations
---------------------------

- ViViT / TimeSformer: best run on a machine with >=1 GPU (8+ GB VRAM). Clip length and model size directly affect memory usage.
- LSTM models: can run on CPU for smaller datasets; GPU speeds up training when sequences / batch sizes grow.

Evaluation and inference
------------------------

- Transformer notebooks include evaluation loops that compute validation loss/accuracy and save best model checkpoints.
- ViViT notebook also contains a `predict_video` helper to run a single video through the model and visualize frames with prediction.
- LSTM notebook includes model evaluation, confusion matrix, and a classification report.

Citations & references
----------------------

- ViViT: Arnab et al., "ViViT: A Video Vision Transformer" (https://arxiv.org/abs/2103.15691)
- TimeSformer: Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?" (https://arxiv.org/abs/2102.05095)

License & data
--------------

This folder contains example notebooks only. Datasets and pre-trained weights are not included. Respect dataset and model licenses when using them. Add a LICENSE file if you plan to publish or share code.

Next steps I can help with
-------------------------

- Create a `requirements.txt` listing exact packages used by each notebook.
- Convert a notebook into a runnable script for non-interactive training.
- Add a short dataset preparation script for a public sign-language dataset (name one: ASL, RWTH-PHOENIX, etc.).

If you want any of the above, tell me which and I'll implement it.
Simple notebooks for sign-language recognition using three approaches:

- ViViT (video transformer) — `ViVit_Fine_Tuning.ipynb`
- TimeSformer (space–time transformer) — `TimeSformer_Fine_Tuning.ipynb`
- LSTM and LSTM with attention (sequence models) — `lstm and lstm with attention.ipynb`

Quick start
-----------

1. Install minimal dependencies (adjust as needed):

```bash
pip install torch torchvision jupyterlab numpy opencv-python
```

2. Start Jupyter and open a notebook:

```bash
jupyter lab
```

3. Edit dataset paths in the notebook and run cells. For transformer notebooks, use short clips (e.g., 8–32 frames).

Dataset layout (simple)
----------------------

Place videos under class folders, for example:

```
dataset/
	train/
		class_A/
			vid1.mp4
		class_B/
			vid2.mp4
```

Notes
-----

- Use a GPU for ViViT/TimeSformer if possible. LSTM notebooks can run on CPU.
- Check each notebook's first cells for exact imports and required packages (some cells may use `timm`, `pytorchvideo`, or `tensorflow`).

Need more?
---------

If you want, I can:
- add a `requirements.txt` with exact packages used by the notebooks
- make a minimal script to run one notebook headless
- add dataset preparation instructions for a specific public dataset

Tell me which and I'll do it next.