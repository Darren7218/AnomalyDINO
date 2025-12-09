# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-level architecture

This project implements a few-shot anomaly detection system named AnomalyDINO, which uses the DINOv2 vision transformer. The core logic is to extract patch-based features from images and use a k-NN search to find anomalies.

- **`run_anomalydino.py`**: The main script for running few-shot anomaly detection experiments. It handles command-line arguments, sets up the model and data, and iterates through different configurations (shots, seeds).
- **`run_anomalydino_batched.py`**: A script for the batched zero-shot anomaly detection scenario.
- **`src/`**: This directory contains the main source code.
    - **`src/backbones.py`**: Defines wrappers for vision transformer models like DINOv2 and standard ViT. It handles model loading, image preparation, and feature extraction.
    - **`src/detection.py`**: Contains the core anomaly detection logic in the `run_anomaly_detection` function. This includes building a memory bank of reference features, performing k-NN search using Faiss, and calculating anomaly scores.
    - **`src/utils.py`**: Provides utility functions for image augmentation (e.g., rotation), processing distance maps, and defining dataset-specific configurations (e.g., which objects require masking or rotation).
    - **`src/post_eval.py`**: Contains functions for evaluating the results of an experiment.
    - **`src/visualize.py`**: Contains functions for creating plots and visualizations.

## Commands

### Installation

To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration with Faiss, a conda environment is recommended.

### Running Experiments

The main script is `run_anomalydino.py`. Here are some common use cases:

- **Run on MVTec dataset with multiple shots and seeds:**
  ```bash
  python run_anomalydino.py --dataset MVTec --shots 1 2 4 8 16 --num_seeds 3 --preprocess agnostic --data_root data/mvtec_anomaly_detection
  ```

- **Run on VisA dataset with multiple shots and seeds:**
  ```bash
  python run_anomalydino.py --dataset VisA --shots 1 2 4 8 16 --num_seeds 3 --preprocess agnostic --data_root data/VisA_pytorch/1cls/
  ```

- **Run a faster inspection on MVTec:**
  ```bash
  python run_anomalydino.py --dataset MVTec --shots 1 --num_seeds 1 --preprocess informed --data_root data/mvtec_anomaly_detection
  ```

- **Run batched zero-shot on MVTec:**
  ```bash
  python run_anomalydino_batched.py --dataset MVTec --data_root data/mvtec_anomaly_detection
  ```

- **Run batched zero-shot on VisA:**
  ```bash
  python run_anomalydino_batched.py --dataset VisA --data_root data/VisA_pytorch/1cls/
  ```

Key command-line arguments in `run_anomalydino.py`:
- `--dataset`: The dataset to use (e.g., `MVTec`, `VisA`).
- `--shots`: A list of the number of reference samples (k-shots).
- `--num_seeds`: The number of repetitions for the experiment.
- `--preprocess`: The preprocessing strategy (`agnostic`, `informed`).
- `--data_root`: The path to the dataset.
- `--model_name`: The backbone model to use (e.g., `dinov2_vits14`).
- `--faiss_on_cpu`: To run kNN search on the CPU.
