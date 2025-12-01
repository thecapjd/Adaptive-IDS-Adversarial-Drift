# Resilience against Adversarial AI in Port Scanning

This repository contains the official implementation of the **Model-Agnostic Adaptive Defense Architecture** presented in the paper:

> *"Resilience against Adversarial AI in Port Scanning: Adaptive Models in Machine Learning and Deep Learning" (2025)*

This research validates an Agent-Centric architecture capable of detecting Concept Drift in network traffic and triggering autonomous adaptation protocols. It benchmarks two learning paradigms: **Static Batch Learning (Random Forest)** versus **Dynamic Incremental Learning (CNN-LSTM)**.

## Repository Structure

The code follows the sequential methodology of the paper:

- **`00_data_harmonization.py`**: ETL pipeline that harmonizes UNSW-NB15 and CIC-IDS-2017 into a unified `master_dataset.csv` (English Schema).
- **`01_baseline_training.py`**: Trains the initial baseline models on benign/normal traffic.
- **`02_adversarial_generation.py`**: Generates the "Mimicry + Low-Rate" adversarial dataset to evade static classifiers.
- **`03_vulnerability_assessment.py`**: Validates the catastrophic failure of static models (RF Recall: 37%, CNN Recall: 3.7%).
- **`04_adaptive_defense_simulation.py`**: **[Main Algorithm]** Implements the Adaptive Agent simulation, demonstrating the **240x speedup** in recovery.
- **`05_visualization_metrics.py`**: Generates Confusion Matrices for the vulnerability analysis.

## Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- scikit-learn, pandas, numpy, matplotlib, seaborn

### Installation

```bash
pip install -r requirements.txt
```

### Reproduction Steps

1. **Data Setup**: Download the raw UNSW-NB15 and CIC-IDS-2017 CSV files and place them in the root folder.

2. **Execute the Pipeline** (in sequential order):

```bash
python 00_data_harmonization.py
python 01_baseline_training.py
python 02_adversarial_generation.py
python 03_vulnerability_assessment.py
python 04_adaptive_defense_simulation.py
```

3. **Output Files Generated**:
   - `master_dataset.csv` - Harmonized training dataset
   - `rf_model.joblib` - Trained Random Forest classifier
   - `cnn_lstm_model.keras` - Trained CNN-LSTM model
   - `scaler.joblib` - Feature scaler (MinMaxScaler)
   - `adversarial_dataset.csv` - Evasive traffic samples
   - `simulation_results.png` - Comparative performance graphs

## Key Results

The simulation confirms that while both paradigms can restore detection fidelity (F1 > 0.99), the **Dynamic Paradigm (CNN-LSTM)** achieves a **240x speedup** in Time-to-Recovery:

| Metric | Static (RF) | Dynamic (CNN-LSTM) | Improvement |
|--------|------------|-------------------|-------------|
| **Time-to-Recovery** | 658s | 2.74s | 240x faster |
| **F1-Score Recovery** | 0.99 | 0.99 | Equivalent |
| **Blind Spot Duration** | 11 min | 2.74s | Eliminates gap |
| **Computational Cost** | O(N) | O(B) | Linear vs Batch |

## Methodology

### Phase 1: Baseline Training
- Random Forest: 100 estimators, balanced tree depth
- CNN-LSTM: Conv1D → LSTM(50) → Dense, trained on 80% data

### Phase 2: Adversarial Generation
- **Statistical Mimicry**: Samples flow characteristics from benign distribution
- **Low-Rate Temporal Evasion**: Increases inter-arrival times by 10x

### Phase 3: Vulnerability Assessment
- Evaluates pre-trained models against adversarial traffic
- Quantifies performance collapse (Recall drops from 99% → 3.7%)

### Phase 4: Adaptive Defense Simulation
- **Drift Detection**: Kolmogorov-Smirnov test (α=0.01) on feature distributions
- **Static Adaptation**: Full retraining on accumulated data (O(N))
- **Dynamic Adaptation**: Incremental fine-tuning on current batch (O(B))

## Visualization

The repository generates comparative plots showing:
- **Panel 1**: Resilience curves (F1-Score recovery trajectory)
- **Panel 2**: Efficiency analysis (Time-to-Recovery comparison)
- **Confusion Matrices**: False positives vs. false negatives per paradigm

## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@article{YourName2025Resilience,
  title={Resilience against Adversarial AI in Port Scanning: 
         Adaptive Models in Machine Learning and Deep Learning},
  author={Your Name},
  year={2025},
  journal={[Journal Name]}
}
```

## Notes

- All experiments use `random_state=42` for reproducibility
- The dataset is harmonized into English schema for cross-domain compatibility
- Batch size for simulation: 50,000 records per iteration
- Drift detection significance level: α = 0.01

## Requirements

See `requirements.txt` for complete dependencies. Key packages:
- TensorFlow/Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- scipy (for KS-test)
- joblib (for model serialization)
---

**Last Updated**: December 2025  
**Paper Status**: Under Review / Accepted at [Journal/Conference Name]
