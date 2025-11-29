# CS229 Project Checklist - EM Adaptive Computation
**Due: December 5, 2024 (Midnight)**  
**Current Date: November 28, 2024**

---

## 📊 Overall Progress Tracker
- [ ] Day 1 (Nov 28) - Core Architecture
- [ ] Day 2 (Nov 29) - Training Pipeline
- [ ] Day 3 (Nov 30) - EM Algorithm
- [ ] Day 4 (Dec 1) - Routers & Baselines
- [ ] Day 5 (Dec 2) - Experiments
- [ ] Day 6 (Dec 3) - Visualization
- [ ] Day 7 (Dec 4) - Report Writing
- [ ] Day 8 (Dec 5) - Final Submission

---

## 🔴 Day 1: Thursday, Nov 28 - CRITICAL FOUNDATION
**Goal: Get core architecture running**

### Feature Caching System (2 hours)
- [ ] Implement FeatureCache class in `algorithms/feature_cache.py`
- [ ] Add methods to extract features from all exits
- [ ] Pre-compute features for training set (50k samples)
- [ ] Pre-compute features for test set (10k samples)
- [ ] Save cached features to disk (use `torch.save`)
- [ ] Test loading cached features

### Configuration (30 mins)
- [ ] Create `configs/default.yaml`
- [ ] Add hyperparameters: learning rates, batch sizes
- [ ] Add lambda values: [0.1, 0.2, 0.5, 1.0]
- [ ] Add EM iterations: 10
- [ ] Add router hidden dimension: 64

---

## 🟠 Day 2: Friday, Nov 29 - TRAINING PIPELINE
**Goal: Train exit classifiers**

### Exit Classifier Training (3 hours)
- [ ] Implement training loop in `training/train_exits.py`
- [ ] Load cached features from disk
- [ ] Create optimizer (SGD, lr=0.01)
- [ ] Train all 4 exit classifiers simultaneously
- [ ] Add progress logging (loss per epoch)
- [ ] Save trained weights to `checkpoints/exits.pth`
- [ ] Verify training converges (loss decreases)

### Basic Evaluation (2 hours)
- [ ] Implement accuracy computation in `training/evaluate.py`
- [ ] Add `evaluate_exits()` function
- [ ] Test Exit 1 accuracy on test set
- [ ] Test Exit 2 accuracy on test set
- [ ] Test Exit 3 accuracy on test set
- [ ] Test Exit 4 accuracy on test set
- [ ] Document all accuracies (should be >70%)

### Sanity Checks (1 hour)
- [ ] Plot training loss curves
- [ ] Verify Exit 4 has highest accuracy
- [ ] Check gradient flow (no NaN values)
- [ ] Save baseline metrics to results file

---

## 🟡 Day 3: Saturday, Nov 30 - EM ALGORITHM
**Goal: Implement core EM routing**

### EM Routing Algorithm (4 hours)
- [ ] Implement EMRouter class in `algorithms/em_routing.py`
- [ ] Implement E-step: compute posteriors P(z|x,y)
- [ ] Add accuracy checking (indicator function)
- [ ] Add cost computation (k/4 normalization)
- [ ] Implement M-step: update priors π_k
- [ ] Add convergence tracking (log-likelihood)
- [ ] Test with lambda=0.5 on small subset
- [ ] Verify EM converges in <10 iterations
- [ ] Run full EM on entire training set

### Router Network (2 hours)
- [ ] Implement Router class in `models/routers.py`
- [ ] Create architecture: Linear(in_dim, 64) + ReLU + Linear(64, 1) + Sigmoid
- [ ] Add forward pass
- [ ] Test with dummy features
- [ ] Verify output is in [0,1] range

### EM Assignment Generation (1 hour)
- [ ] Run EM with lambda=0.1, save assignments
- [ ] Run EM with lambda=0.5, save assignments
- [ ] Run EM with lambda=1.0, save assignments
- [ ] Save assignments to `checkpoints/em_assignments/`
- [ ] Visualize assignment distributions (histogram)

---

## 🟢 Day 4: Sunday, Dec 1 - ROUTERS & BASELINES
**Goal: Train routers and implement baselines**

### Router Training (3 hours)
- [ ] Implement training loop in `training/train_routers.py`
- [ ] Load EM assignments from disk
- [ ] Create binary labels (exit ≤ k)
- [ ] Train 4 routers (one per exit)
- [ ] Use BCE loss with Adam optimizer
- [ ] Add learning rate: 0.001
- [ ] Save trained routers to `checkpoints/routers/`
- [ ] Test router predictions on validation set

### Baseline Implementations (3 hours)
- [ ] Create baseline functions in `experiments/baselines.py`
- [ ] Implement static baseline (always use Exit 4)
- [ ] Implement random routing baseline
- [ ] Implement confidence-based (entropy < threshold)
- [ ] Implement oracle baseline (optimal per-sample routing)
- [ ] Evaluate all baselines on test set
- [ ] Save baseline results to `results/baselines.json`

---

## 🔵 Day 5: Monday, Dec 2 - COMPREHENSIVE EXPERIMENTS
**Goal: Run all experiments and collect results**

### Main Entry Point (1 hour)
- [ ] Create `main.py` with argument parser
- [ ] Add mode: train_exits
- [ ] Add mode: run_em
- [ ] Add mode: train_routers
- [ ] Add mode: evaluate
- [ ] Test end-to-end pipeline
- [ ] Document usage in README

### Full Pipeline Execution (3 hours)
- [ ] Run complete pipeline with lambda=0.1
- [ ] Run complete pipeline with lambda=0.5
- [ ] Run complete pipeline with lambda=1.0
- [ ] Evaluate EM routing vs static baseline
- [ ] Evaluate EM routing vs random baseline
- [ ] Evaluate EM routing vs confidence baseline
- [ ] Evaluate EM routing vs oracle baseline
- [ ] Save all results to `results/experiments.json`

### Ablation Studies (2 hours)
- [ ] Implement ablations in `experiments/ablations.py`
- [ ] Study: EM iterations (5, 10, 20)
- [ ] Study: Router depth (1-layer vs 2-layer)
- [ ] Study: Hard vs soft EM assignments
- [ ] Save ablation results to `results/ablations.json`

---

## 🟣 Day 6: Tuesday, Dec 3 - VISUALIZATION & ANALYSIS
**Goal: Create all figures and analyze results**

### Visualization Implementation (3 hours)
- [ ] Implement plotting functions in `experiments/visualization.py`
- [ ] Create Pareto frontier plot (accuracy vs FLOPs)
- [ ] Create exit distribution heatmap (per-class)
- [ ] Create EM convergence plot (log-likelihood)
- [ ] Create router threshold sensitivity plot
- [ ] Create comparison bar chart (all baselines)
- [ ] Save all figures to `figures/`

### Results Analysis (2 hours)
- [ ] Create results table (accuracy, avg FLOPs, speedup)
- [ ] Compute statistical significance (t-tests)
- [ ] Analyze per-class routing patterns
- [ ] Identify failure cases
- [ ] Document key findings in `RESULTS.md`

### Code Cleanup (1 hour)
- [ ] Add docstrings to all classes
- [ ] Add docstrings to all functions
- [ ] Remove commented-out code
- [ ] Add type hints
- [ ] Run code formatter (black)
- [ ] Test reproducibility with fixed seed

---

## 📝 Day 7: Wednesday, Dec 4 - FINAL REPORT
**Goal: Write paper and prepare submission**

### Report Writing (6 hours)
- [ ] Introduction (1 hour)
  - [ ] Motivation
  - [ ] Problem statement
  - [ ] Our contribution
- [ ] Related Work (30 mins)
  - [ ] Multi-exit networks
  - [ ] EM algorithm applications
- [ ] Methods (1.5 hours)
  - [ ] Architecture description
  - [ ] EM formulation
  - [ ] Router training
- [ ] Experiments (1.5 hours)
  - [ ] Setup description
  - [ ] Main results
  - [ ] Ablation studies
- [ ] Discussion & Conclusion (1 hour)
  - [ ] Key findings
  - [ ] Limitations
  - [ ] Future work
- [ ] Abstract (30 mins)
  - [ ] Concise summary
- [ ] References (add proper citations)

### Final Checks (2 hours)
- [ ] Run entire pipeline from scratch
- [ ] Verify all figures render correctly
- [ ] Check all tables formatted properly
- [ ] Update README with final results
- [ ] Create requirements.txt
- [ ] Test on clean environment
- [ ] Zip code for submission

---

## 🎯 Day 8: Thursday, Dec 5 (DEADLINE) - SUBMISSION
**Goal: Final polish and submit before midnight**

### Morning: Final Review (3 hours)
- [ ] Proofread entire paper (grammar, typos)
- [ ] Check figure captions
- [ ] Verify table numbers
- [ ] Check equation formatting
- [ ] Verify all citations present
- [ ] Get feedback from classmate (if possible)
- [ ] Make final revisions

### Afternoon: Submission Package (2 hours)
- [ ] Create submission folder
- [ ] Include: Final report PDF
- [ ] Include: Code (zipped)
- [ ] Include: README with instructions
- [ ] Include: requirements.txt
- [ ] Include: Sample outputs
- [ ] Test code on fresh clone
- [ ] Verify everything runs

### Evening: Submit (BEFORE MIDNIGHT)
- [ ] Upload to Canvas/submission portal
- [ ] Verify file uploaded correctly
- [ ] Check submission confirmation
- [ ] Backup to Google Drive
- [ ] Backup to GitHub (private repo)
- [ ] **CELEBRATE!** 🎉

---

## 📌 Critical Reminders

### Must-Have Results
- [ ] EM routing beats random baseline
- [ ] EM routing beats confidence baseline
- [ ] Clear Pareto frontier shown
- [ ] At least 3 lambda values evaluated

### Time-Savers
- [ ] Use CPU + feature caching (don't need GPU)
- [ ] Start long experiments overnight
- [ ] Don't over-optimize - working > perfect
- [ ] Focus on core novelty (EM routing)

### Emergency Backup Plans
- [ ] If EM doesn't converge: simplify formulation
- [ ] If routers don't train: use EM assignments directly
- [ ] If experiments take too long: use smaller subset
- [ ] If time runs short: prioritize main results over ablations

---

## 📊 Daily Time Budget (8 hours/day)
- **Implementation:** 50% (4 hours)
- **Experiments:** 30% (2.4 hours)
- **Analysis/Writing:** 20% (1.6 hours)

**Good luck! You've got this! 💪**
