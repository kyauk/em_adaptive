# EM-Based Adaptive Computation in Multi-Exit Neural Networks

CS229 Final Project - Stanford University

## Motivation: Real-Time Medical Imaging for Surgical Robotics

Modern surgical robotics systems, rely on real-time computer vision for critical tasks including tissue classification, instrument tracking, and anatomical segmentation. During minimally invasive procedures, these systems must process high-resolution video streams at 30+ FPS while running on embedded hardware with strict power and thermal constraints.

**The Challenge**: Deep neural networks achieve state-of-the-art accuracy on medical imaging tasks but require uniform computation for all inputs—even when many frames contain simple, low-risk scenes that could be classified with minimal processing. This computational inefficiency limits deployment of advanced vision models in the operating room.

**Existing Approaches**: Prior work (BranchyNet, MSDNet) addresses this via local heuristics: exit early when entropy is low or confidence is high. However, these methods require manual threshold tuning per layer and lack a principled framework for trading off accuracy vs. efficiency globally.

**Our Contribution**: We reframe adaptive computation as a probabilistic inference problem. Instead of hand-crafting exit rules, we discover the optimal distribution of samples across network depths using Expectation-Maximization (EM). This shift from heuristics to principled inference enables:
1. **Global Optimization**: EM jointly considers all exits when assigning samples
2. **Automatic Tuning**: The λ parameter provides a single knob for the accuracy-efficiency tradeoff
3. **Theoretical Grounding**: Latent variable formulation with convergence guarantees

## Abstract

This project presents a novel approach to adaptive computation in neural networks by formulating exit point selection as a latent variable inference problem solved through Expectation-Maximization (EM). **Unlike existing methods that rely on local heuristics (entropy thresholds, confidence scores), we take a principled probabilistic view: treating the distribution of samples across network exits as a latent variable to be discovered through global inference.** This shift from hand-crafted rules to data-driven distribution discovery enables our method to automatically balance accuracy and efficiency without manual tuning of per-layer thresholds. While our experiments use CIFAR-10 as a controlled testbed, the methodology directly applies to medical imaging scenarios where computational budgets are constrained and deployment reliability is critical.

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/kyauk/em_adaptive.git
cd em_adaptive

# Install dependencies
pip install torch torchvision pyyaml matplotlib tqdm

# Download CIFAR-10 (automatic on first run)
```

### Running Experiments

**Option 1: Full Automated Pipeline**
```bash
# Run complete Pareto sweep (trains for multiple lambda values)
python experiments/pareto_sweep.py

# Visualize results
python experiments/visualization.py
```

**Option 2: Manual Step-by-Step**
```bash
# 1. Train exit classifiers
python main.py --mode train --train_target train_exits

# 2. Train routers with specific lambda
python main.py --mode train --train_target train_routers --lambda_val 0.05

# 3. Evaluate all methods
python main.py --mode evaluate --method all --threshold 0.5

# 4. Evaluate specific method
python main.py --mode evaluate --method em_routing --threshold 0.7
```

## System Architecture

### Multi-Exit ResNet-18

We use a ResNet-18 backbone trained from scratch on CIFAR-10 (32×32 native resolution) with four exit points:

| Exit | Location | Features | Relative Cost |
|------|----------|----------|---------------|
| Exit 1 | Post-Layer1 | 64-dim | 25% |
| Exit 2 | Post-Layer2 | 128-dim | 50% |
| Exit 3 | Post-Layer3 | 256-dim | 75% |
| Exit 4 | Post-Layer4 | 512-dim | 100% |

**Design Rationale**: This architecture mirrors the computational constraints of embedded surgical systems, where early exits correspond to low-latency, low-power inference pathways suitable for routine frame processing, while deeper exits handle complex anatomical structures or critical decision points.

### Training Pipeline

#### Phase 1: Exit Classifier Training

Train lightweight classification heads at each exit point while keeping the backbone frozen:

```
L_total = Σ(k=1 to 4) CrossEntropy(f_k(x), y)
```

Each classifier consists of global average pooling followed by a single linear layer.

#### Phase 2: EM-Based Assignment Discovery

We formulate exit selection as a latent variable problem where z_i ∈ {1,2,3,4} represents the exit assignment for sample i.

**E-Step:** Compute posterior probabilities:
```
P(z_i = k | x_i, y_i) ∝ π_k · P(correct | x_i, exit_k) · exp(-λ · cost_k)
```

**M-Step:** Update priors:
```
π_k = (1/N) Σ_i P(z_i = k | x_i, y_i)
```

Where:
- π_k: prior probability of using exit k
- λ: accuracy-efficiency tradeoff parameter (critical for safety-constrained systems)
- cost_k = k/4: normalized computational cost

**Medical Relevance**: The λ parameter allows surgeons to tune the system's operating point based on procedure criticality—higher λ during high-risk phases (near critical structures), lower λ during routine manipulation.

#### Phase 3: Router Network Training

Train compact neural networks to predict exit decisions:

```
R_k: Features_k → [0,1]
L_router = Σ_i BCE(R_k(F_i^k), 1[assignment_i = k])
```

Each router learns a binary classification task: "Should sample exit at exit k?"

#### Phase 4: Inference

Samples traverse the network until a router signals to exit:

```python
for k in [1, 2, 3, 4]:
    features_k = backbone[:k](x)
    if router_k(features_k) > threshold:
        return classifier_k(features_k)
return classifier_4(features_4)  # fallback to full network
```

## Evaluation Methods

### Implemented Baselines

1. **Standard ResNet-18**: Full network (accuracy upper bound)
2. **Fixed Exits**: Always use specific exit (1, 2, 3, or 4)
3. **Random Routing**: Uniform random exit selection
4. **BranchyNet**: Entropy-based early exiting
5. **Oracle**: Perfect per-sample routing (efficiency upper bound)
6. **EM Routing**: Our proposed method

### Metrics

- Classification Accuracy (safety-critical)
- Average Computational Cost (normalized FLOPs)
- Accuracy-Cost Pareto Frontier

## Experimental Protocol

### Hyperparameters

```python
# Exit Classifier Training
learning_rate = 0.01
epochs = 30
batch_size = 128

# EM Algorithm
iterations = 5
lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]

# Router Training
learning_rate = 0.001
epochs = 20
hidden_dim = 64

# Inference
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

### Pareto Sweep

The `experiments/pareto_sweep.py` script automates:
1. Loop over lambda values (training)
2. Train routers for each lambda
3. Loop over threshold values (inference)
4. Evaluate and save (accuracy, cost) pairs
5. Save results to `results/pareto_results.json`

## Implementation Details

### Feature Caching Strategy

To accelerate experimentation, we precompute and cache all intermediate features:

```python
features[sample_id] = {
    'layer_k': backbone.layer_k(x).detach()
    for k in [1, 2, 3, 4]
}
```

This reduces training time from O(epochs × samples × depth) to O(samples × depth) + O(epochs × samples).

### Device Management

Code automatically selects the best available device:
- CUDA (cloud GPU)
- MPS (Apple Silicon)
- CPU (fallback)

## Project Structure

```
├── main.py                       # Main entry point with CLI
├── models/
│   ├── multi_exit_resnet.py     # Multi-exit architecture
│   └── routers.py               # Router networks
├── algorithms/
│   ├── em_routing.py            # EM algorithm implementation
│   └── feature_cache.py         # Feature caching utilities
├── training/
│   ├── train_exits.py           # Exit classifier training
│   └── train_routers.py         # Router training
├── experiments/
│   ├── evaluation.py            # All evaluation methods
│   ├── pareto_sweep.py          # Automated parameter sweep
│   └── visualization.py         # Result plotting
├── dataloader.py                # CIFAR-10 data loading
└── checkpoints/                 # Saved models
```

## Expected Contributions

1. **Theoretical**: First formulation of multi-exit routing as latent variable inference
2. **Empirical**: Demonstration that unsupervised routing discovery can match supervised approaches
3. **Practical**: Efficient training methodology using frozen backbones and feature caching
4. **Clinical Relevance**: Framework enables tunable accuracy-latency tradeoffs for safety-critical medical applications

## Translation to Medical Imaging

While this work uses CIFAR-10 for controlled experimentation, the methodology directly extends to surgical vision:

**Potential Applications:**
- **Tissue Classification**: Route simple muscle/fat frames through shallow networks, reserve deep computation for complex tumor margins
- **Instrument Tracking**: Early exit on clear views, deep processing when instruments are partially occluded
- **Anatomical Segmentation**: Adaptive computation based on anatomical complexity

**Safety Considerations**: The λ parameter provides a principled mechanism for trading accuracy vs. latency, enabling surgeons to configure the system for procedure-specific risk tolerance.

## Reproducibility

All experiments are designed to run on CPU with modest memory requirements through feature caching. Full GPU support is included for cloud deployment.

**Reproducibility Checklist:**
- ✅ Fixed random seeds
- ✅ Deterministic data loading
- ✅ Complete hyperparameter documentation
- ✅ Cached features for consistency

## Dependencies

```
torch >= 1.12.0
torchvision >= 0.13.0
numpy >= 1.21.0
matplotlib >= 3.3.0
pyyaml >= 5.4.0
tqdm >= 4.62.0
```

## Future Work

- Extension to Vision Transformers (ViT)
- Adaptive sampling policies for object detection
- Transfer learning to medical imaging datasets (EndoVis, Cholec80)
- Integration with real-time surgical video streams
- Reinforcement learning refinement based on surgeon feedback

## References

[1] Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). BranchyNet: Fast inference via early exiting from deep neural networks.

[2] Huang, G., et al. (2018). Multi-scale dense networks for resource efficient image classification.

[3] Scardapane, S., et al. (2020). Group sparse regularization for deep neural networks.

## Acknowledgments

CS229 Teaching Staff, Stanford University, Fall 2025

## License

MIT License - See LICENSE file for details