# EM-Based Adaptive Computation in Multi-Exit Neural Networks

CS229 Final Project - Stanford University

## Abstract

This project presents a novel approach to adaptive computation in neural networks by formulating exit point selection as a latent variable inference problem solved through Expectation-Maximization (EM). Unlike existing methods that rely on predetermined confidence thresholds or supervised gating mechanisms, we discover optimal routing policies through unsupervised probabilistic inference that jointly optimizes for classification accuracy and computational efficiency.

## Introduction

Deep neural networks apply uniform computation to all inputs regardless of complexity, leading to computational inefficiency in real-time applications. While multi-exit architectures have been proposed to address this issue, determining optimal routing policies remains challenging. Current approaches rely on local decision rules, whereas we propose a global optimization framework that discovers the optimal distribution of samples across network exits.

## Technical Approach

### System Architecture

We augment a pretrained ResNet-18 with four early exit classifiers positioned after each residual block group. The backbone network remains frozen to eliminate gradient interference and isolate the routing problem from representation learning.

**Exit Points:**
- Exit 1: Post-Layer1 (64-dimensional features, ~25% FLOPs)
- Exit 2: Post-Layer2 (128-dimensional features, ~50% FLOPs)  
- Exit 3: Post-Layer3 (256-dimensional features, ~75% FLOPs)
- Exit 4: Post-Layer4 (512-dimensional features, 100% FLOPs)

### Training Pipeline

#### Phase 1: Exit Classifier Training

We train lightweight classification heads at each exit point while keeping the backbone frozen:

```
L_total = Σ(k=1 to 4) CrossEntropy(f_k(x), y)
```

Each classifier consists of global average pooling followed by a single linear layer mapping features to class logits.

#### Phase 2: EM-Based Assignment Discovery

We formulate exit selection as a latent variable problem where z_i ∈ {1,2,3,4} represents the exit assignment for sample i.

**E-Step:** Compute posterior probabilities for each sample-exit pair:
```
P(z_i = k | x_i, y_i) ∝ π_k · 1[f_k(x_i) = y_i] · exp(-λ · cost_k)
```

Where:
- π_k represents the prior probability of using exit k
- 1[·] is an indicator function for correct classification
- cost_k = k/4 normalizes computational cost
- λ controls the accuracy-efficiency tradeoff

**M-Step:** Update parameters:
```
π_k = (1/N) Σ_i P(z_i = k | x_i, y_i)
```

#### Phase 3: Router Network Training

We train compact neural networks to predict exit decisions based on intermediate features:

```
R_k: F_k → [0,1]
```

Each router is trained using binary cross-entropy loss with EM assignments as supervision:
```
L_router = Σ_i BCE(R_k(features_i^k), 1[EM_assignment_i ≤ k])
```

#### Phase 4: Inference

During inference, samples traverse the network until a router signals to exit:

```python
def forward(x):
    for k in [1, 2, 3, 4]:
        features_k = backbone[:k](x)
        if router_k(features_k) > threshold:
            return classifier_k(features_k)
    return classifier_4(features_4)
```

## Experimental Protocol

### Datasets
- Primary: CIFAR-10 (50,000 training, 10,000 test images)
- Secondary: CIFAR-100 (transfer learning evaluation)

### Evaluation Metrics
- Classification accuracy
- Average computational cost (FLOPs)
- Per-class exit distribution
- Accuracy-efficiency Pareto frontier

### Baselines
1. **Static Network**: Full ResNet-18 (upper bound accuracy)
2. **Random Routing**: Uniform random exit selection
3. **Confidence-Based**: Entropy thresholding (BranchyNet-style)
4. **Oracle**: Perfect difficulty-aware routing (upper bound efficiency)

### Ablation Studies
- Impact of λ on accuracy-compute tradeoff
- EM convergence analysis
- Router architecture depth
- Hard vs. soft EM assignments

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

### Hyperparameters
- Exit classifier learning rate: 0.01
- Router learning rate: 0.001
- EM iterations: 10
- λ range: [0.1, 1.0]
- Router hidden dimension: 64

## Expected Contributions

1. **Theoretical**: First formulation of multi-exit routing as latent variable inference
2. **Empirical**: Demonstration that unsupervised routing discovery can match supervised approaches
3. **Practical**: Efficient training methodology using frozen backbones and feature caching

## Project Timeline

**Week 1:**
- Days 1-2: Architecture implementation and exit classifier training
- Days 3-4: EM algorithm implementation and debugging
- Days 5-6: Router network training
- Day 7: Initial evaluation

**Week 2:**
- Days 8-9: Baseline implementations
- Days 10-11: Comprehensive experiments
- Day 12: Ablation studies
- Days 13-14: Analysis and documentation

## Reproducibility

All experiments are designed to run on CPU with modest memory requirements through aggressive feature caching. Code will be made available at project completion.

## Dependencies

- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- Scikit-learn >= 0.24.0
- Matplotlib >= 3.3.0

## Project Structure

```
├── models/
│   ├── multi_exit_resnet.py
│   ├── routers.py
│   └── exits.py
├── algorithms/
│   ├── em_routing.py
│   └── feature_cache.py
├── training/
│   ├── train_exits.py
│   ├── train_routers.py
│   └── evaluate.py
├── experiments/
│   ├── ablations.py
│   ├── baselines.py
│   └── visualization.py
├── configs/
│   └── default.yaml
└── main.py
```

## Future Work

This framework naturally extends to reinforcement learning, where discovered policies can be refined through environmental interaction. Additionally, the approach could be adapted for other architectures (Vision Transformers) and tasks (object detection, semantic segmentation).

## References

[1] Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). BranchyNet: Fast inference via early exiting from deep neural networks.

[2] Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., & Weinberger, K. Q. (2018). Multi-scale dense networks for resource efficient image classification.

[3] Scardapane, S., Comminiello, D., Hussain, A., & Uncini, A. (2020). Group sparse regularization for deep neural networks.

## Acknowledgments

CS229 Teaching Staff, Stanford University, Fall 2024