# Running AnomalyDINO on MVTec AD Dataset

This document outlines the steps to run AnomalyDINO on the MVTec Anomaly Detection (AD) dataset.

## Prerequisites

1.  **Virtual Environment:** Create and activate a Python virtual environment.
    ```bash
    python -m venv .venvAnomalyDINO
    source .venvAnomalyDINO/bin/activate
    ```

2.  **Install Dependencies:** Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Dataset Preparation:** Download and prepare the MVTec-AD dataset from its official source. The default data root for MVTec-AD is `data/mvtec_anomaly_detection`. Ensure your data is organized as follows:
    ```
    your_data_root
    ├── object1
    │   ├── ground_truth        # anomaly annotations per anomaly type
    │   │   ├── anomaly_type1
    │   │   ├── ...
    │   ├── test                # test images per anomaly type & 'good'
    │   │   ├── anomaly_type1
    │   │   ├── ...
    │   │   └── good
    │   └── train               # train/reference images (without anomalies)
    │       └── good
    ├── object2
    │   ├── ...
    ```

## Usage

### Few-shot Anomaly Detection

To perform few-shot anomaly detection, run the `run_anomalydino.py` script with the following command (example for MVTec with all considered shots, three repetitions, and agnostic preprocessing):

```bash
python run_anomalydino.py --dataset MVTec --shots 1 2 4 8 16 --num_seeds 3 --preprocess agnostic --data_root data/mvtec_anomaly_detection
```

For a faster inspection, you can use:

```bash
python run_anomalydino.py --dataset MVTec --shots 1 --num_seeds 1 --preprocess informed --data_root data/mvtec_anomaly_detection
```

### Batched-Zero-Shot Anomaly Detection

To reproduce the results in the *batched* zero-shot scenario, run the `run_anomalydino_batched.py` script:

```bash
python run_anomalydino_batched.py --dataset MVTec --data_root data/mvtec_anomaly_detection
```
