# Edge Shield: Stateful Security Architecture for IoT Networks

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Edge Shield** is a lightweight, resource-constrained intrusion detection architecture engineered specifically for edge routers and IoT networks. It bypasses the severe computational overhead of Deep Learning (DNNs) by combining spatio-temporal traffic feature extraction, payload entropy analysis, and a stateful 5-tuple flow caching protocol.

## Architecture Overview

Traditional enterprise firewalls rely on Deep Packet Inspection (DPI) or multi-megabyte machine learning models. Edge Shield is designed to operate on legacy router silicon (e.g., ARM Cortex-A53 microprocessors) by exploiting the physical transmission mechanics of botnets rather than semantic payload inspection.

1. **Spatio-Temporal Correlation:** Detects automated botnet floods (like Mirai) by measuring Jitter Variance ($\sigma^2$) and Packet Inter-Arrival Time (IAT).
2. **Structural Randomness:** Analyzes payload Shannon Entropy to detect compressed exfiltration or encrypted malware streams.
3. **Stateful Flow Cache:** Employs an $O(1)$ 5-tuple cache lookup. Deep inference is executed only on the initial packet of a flow. Subsequent packets are handled directly by memory, bypassing CPU bottlenecks during volumetric DDoS events.

## Experimental Results

The model was evaluated using a 3.6 million record extract from the empirical **UNSW-2018 Bot-IoT Dataset** utilizing Stratified 5-Fold Cross-Validation.

| Architecture | F1-Score (95% CI) | Memory Footprint | Inference Latency |
| :--- | :--- | :--- | :--- |
| Random Forest | [0.973, 0.977] | 234.82 KB | 107.34 µs |
| XGBoost | [0.976, 0.978] | 129.22 KB | 39.23 µs |
| DNN (MLP) | [0.956, 0.964] | 86.66 KB | 6.11 µs |
| **Edge Shield (Ours)** | **[0.967, 0.971]** | **3.51 KB** | **1.69 µs** |

*Hardware Constraints:* By restricting the Decision Tree depth to exactly $D=5$, execution occurs in mathematically bounded $O(1)$ constant time. In empirical replays, the 5-Tuple cache successfully reduced active AI computation workload by **84.3%**.

## Repository Structure

*   `train.py`: The cross-validation pipeline. Includes in-fold SMOTE balancing to prevent data leakage and outputs exact hardware benchmarking metrics.
*   `inference.py`: The live simulated inference engine containing the Algorithm 1 Cache Logic. 
*   `manuscript.html`: The complete, mathematically formalized IEEE-style research paper detailing the algorithmic complexity proofs and feature distributions.

## Usage

1. Clone the repository and install dependencies:
   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn



   ## Pre-Trained Weights (Kaggle)
Due to the immense size of the cross-domain dataset (3.6+ million rows combining UNSW-2018 and CIC-IoT-2023), the champion model was trained via Kaggle kernels. 

* The `train.py` script is provided for strict academic reproducibility, demonstrating the memory downcasting and leak-free SMOTE balancing pipeline.
* The production-ready `edge_shield_model.pkl` is already included in this repository. You can execute `inference.py` immediately without needing to re-train the architecture or download the raw datasets.
