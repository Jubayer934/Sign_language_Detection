# 🤟 Sign Language Recognition (Video Classification)

## 📝 Description
This repository contains a collection of deep learning implementations for **Sign Language Recognition** or **Video Sign Language Detection**. The project explores several modern and classic architectures to classify signs from video data, focusing on both sequence modeling (using extracted features) and end-to-end video classification.

***

## 🧠 Models and Architectures Explored
The repository features three distinct approaches to tackling the sign language recognition problem:

| Approach | Model(s) | Key Feature | Notebook |
| :--- | :--- | :--- | :--- |
| **Sequence Modeling** | **LSTM** & **LSTM with Attention** | Processes sequential features (e.g., MediaPipe keypoints) for classification. | `lstm and lstm with attention.ipynb` |
| **Video Transformer** | **TimeSformer** (Temporal Shift Module) | An efficient Vision Transformer adapted for video data via temporal attention. | `TimeSformer_Fine_Tuning.ipynb` |
| **Video Transformer** | **ViVit** (Vision Transformer for Video) | Directly applies a Transformer architecture across spatial and temporal dimensions of video. | `ViVit_Fine_Tuning.ipynb` |

***

## 📂 Repository Contents
The main codebase is contained in the following Jupyter Notebooks:

| File Name | Purpose |
| :--- | :--- |
| **`lstm and lstm with attention.ipynb`** | Implements a **Long Short-Term Memory (LSTM)** network and a model with an **Attention Mechanism**. This notebook typically handles data preparation (like feature extraction using tools like MediaPipe) and trains the sequence models. |
| **`TimeSformer_Fine_Tuning.ipynb`** | Shows the process for fine-tuning a pre-trained **TimeSformer** model on the sign language video dataset for high-performance video classification. |
| **`ViVit_Fine_Tuning.ipynb`** | Demonstrates the fine-tuning of a pre-trained **ViVit** (Vision Transformer for Video) model for robust end-to-end sign language video classification. |

***

## 🚀 Getting Started

### Prerequisites
1.  **Dataset:** You will need a structured dataset of sign language videos (e.g., in a folder where each sub-folder represents a class/sign).
2.  **Environment:** Python 3.x is required.
3.  **Hardware:** A modern **GPU** (especially for TimeSformer and ViVit fine-tuning) is highly recommended for faster training times.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPO_URL>
    cd <YOUR_REPO_NAME>
    ```

2.  **Install dependencies:**
    The exact requirements vary per notebook, but you will need packages like `tensorflow`, `keras`, `torch`, `transformers`, `mediapipe`, and `opencv-python`. A good starting point is:
    ```bash
    pip install tensorflow keras torch transformers mediapipe opencv-python
    ```

### Usage
1.  Place your sign language video dataset into the appropriate directory structure as specified in the notebooks.
2.  Open any of the `.ipynb` files in **Google Colab** (many of these notebooks are designed to run there) or a local **Jupyter** environment.
3.  Run the cells sequentially, making sure to adjust the dataset paths and hyper-parameters as needed for your specific task.

***

## 🛠️ Data Preprocessing Note
For the **LSTM** models, the videos are typically processed to extract sequential features (e.g., hand and pose landmarks/keypoints) before training. For the **TimeSformer** and **ViVit** models, preprocessing involves transforming video frames into the required input tensor format for the Vision Transformers.