# EM-Based Adaptive Computation in Multi-Exit Neural Networks

**CS229 Final Project - Stanford University**

## Abstract

This project presents a novel approach to adaptive computation in deep neural networks by formulating exit point selection as a latent variable inference problem. Unlike existing methods such as BranchyNet, which rely on local heuristics (e.g., entropy thresholds) to determine early exits, our method employs the Expectation-Maximization (EM) algorithm to discover the optimal global distribution of samples across network depths. By treating the exit assignment as a latent variable, we derive a principled objective that balances classification accuracy with computational cost through a single tunable parameter ($\lambda$). We implement this framework on a Multi-Exit ResNet-18 architecture and demonstrate that it achieves a superior accuracy-efficiency trade-off compared to standard baselines, reaching **89% accuracy at 75% of the computational cost** of the full network on CIFAR-10. While demonstrated on standard benchmarks, this methodology is motivated by real-time constraints in surgical robotics, where resource-efficient inference is critical for safety and performance.

## 1. Motivation: Real-Time Surgical Vision

Modern surgical robotics systems rely on real-time computer vision for tasks such as instrument tracking and tissue segmentation. These systems operate under strict constraints:
*   **Latency:** Processing must occur at 30+ FPS to provide seamless feedback.
*   **Compute:** Hardware is often embedded and power-constrained.
*   **Variability:** Input complexity varies significantly; many frames are simple (e.g., clear view of tools), while others are complex (e.g., occlusion, smoke).

Standard deep networks apply uniform computation to all inputs, which is inefficient. Our EM-based routing adapts the computational effort to the difficulty of the input, processing simple frames with shallow sub-networks and reserving deep computation for complex cases.

## 2. Methodology

### 2.1 Multi-Exit Architecture
We modify a standard **ResNet-18** backbone by attaching lightweight "Exit Classifiers" at intermediate stages.
*   **Exit 1:** After Layer 1 (Cost: ~25%)
*   **Exit 2:** After Layer 2 (Cost: ~50%)
*   **Exit 3:** After Layer 3 (Cost: ~75%)
*   **Exit 4:** Final Output (Cost: 100%)

### 2.2 EM Routing Algorithm
We formulate the routing problem probabilistically:
*   **Latent Variable ($z$):** The optimal exit for a given sample.
*   **E-Step (Inference):** Estimate the posterior probability of the optimal exit given the input and current model parameters. This considers both the likelihood of correct classification and the computational cost penalty ($\lambda$).
*   **M-Step (Learning):** Update the "Router" networks to predict these posterior probabilities from the intermediate features.

This approach allows the system to learn complex routing policies that go beyond simple confidence thresholding.

## 3. Project Structure

The codebase is organized as follows:

```
.
├── main.py                       # Central entry point for training and evaluation
├── algorithms/
│   ├── em_routing.py            # Implementation of the EM algorithm (E-Step/M-Step)
│   └── feature_cache.py         # Utilities for caching intermediate features
├── models/
│   ├── multi_exit_resnet.py     # ResNet-18 with intermediate exit heads
│   └── routers.py               # Lightweight router networks
├── training/
│   ├── train_exits.py           # Pipeline for training exit classifiers
│   └── train_routers.py         # Pipeline for training routers via EM
├── experiments/
│   ├── pareto_sweep.py          # Script for sweeping lambda/thresholds
│   ├── visualization.py         # Plotting tools for Pareto frontiers
│   └── evaluation.py            # Comprehensive evaluation metrics
└── configs/                      # Configuration files
```

## 4. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kyauk/em_adaptive.git
    cd em_adaptive
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data:**
    The CIFAR-10 dataset will be automatically downloaded to `./cifar-10-batches-py` on the first run.

## 5. Usage

### 5.1 Training
The training process is two-staged: first training the exits, then the routers.

```bash
# 1. Train Exit Classifiers (Frozen Backbone)
python main.py --mode train --train_target train_exits

# 2. Train Routers (using EM with a specific lambda)
python main.py --mode train --train_target train_routers --lambda_val 1.5
```

### 5.2 Evaluation
Evaluate the model using various baselines and the proposed EM method.

```bash
# Run full evaluation (ResNet, BranchyNet, EM Routing, Oracle)
python main.py --mode evaluate --method all --lambda_val 1.5 --threshold 0.6
```

### 5.3 Reproducing Results (Pareto Sweep)
To generate the full accuracy-cost trade-off curve (Pareto Frontier):

```bash
# 1. Run the parameter sweep (trains routers for multiple lambdas)
python -m experiments.pareto_sweep

# 2. Generate visualization plots
python -m experiments.visualization
```
This will produce `results/pareto_plot.png`, comparing EM Routing against the BranchyNet baseline.

## 6. Results

Our experiments demonstrate that EM Routing effectively trades off computation for accuracy.

*   **Key Result:** EM Routing achieves **89.0% Accuracy** at **0.75 Normalized Cost**.
*   **Comparison:** The method outperforms random routing and shows competitive performance to the strong BranchyNet baseline in high-accuracy regimes.
*   **Pareto Frontier:** The generated plots confirm that EM Routing pushes the Pareto frontier of efficient inference.

## 7. Future Work

*   **Real-World Deployment:** Integrating this framework into a real-time surgical video pipeline.
*   **Vision Transformers:** Extending the multi-exit paradigm to ViT architectures.
*   **Dynamic Lambda:** Adapting $\lambda$ in real-time based on system load or surgical context.

## References

1.  Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). **BranchyNet: Fast inference via early exiting from deep neural networks**. *ICPR*.

---
*Author: Jason Kyauk*