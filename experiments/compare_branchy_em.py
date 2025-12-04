"""
Compare BranchyNet vs EM Routing across thresholds and lambdas.
Generates a Pareto plot with both methods.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import json
import os
from main import setup_model, setup_routers, get_dataloader, Config
from training.train_routers import train_routers
from experiments.evaluation import Evaluator

def run_comparison():
    # Parameters
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lambda_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Load model and data
    model = setup_model(load_exits=True)
    evaluator = Evaluator(model)
    _, test_loader = get_dataloader()
    
    # 1. BranchyNet sweep (no lambda dependency)
    print("\n=== BranchyNet Sweep ===")
    branchy_results = []
    for thresh in thresholds:
        result = evaluator.eval_branchynet(test_loader, threshold=thresh)
        branchy_results.append({
            'threshold': thresh,
            'accuracy': result['accuracy'],
            'cost': result['cost']
        })
    
    # 2. EM Routing sweep (different lambdas)
    print("\n=== EM Routing Sweep ===")
    em_results = []
    for lam in lambda_vals:
        print(f"\n--- Training routers with Lambda={lam} ---")
        train_routers(lambda_val=lam)
        
        # Reload routers and evaluate at different thresholds
        routers = setup_routers(load_routers=True)
        for thresh in thresholds:
            result = evaluator.eval_em_routing(test_loader, routers, threshold=thresh)
            em_results.append({
                'lambda': lam,
                'threshold': thresh,
                'accuracy': result['accuracy'],
                'cost': result['cost']
            })
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_results.json', 'w') as f:
        json.dump({'branchynet': branchy_results, 'em': em_results}, f, indent=2)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # BranchyNet (single line)
    branchy_costs = [r['cost'] for r in branchy_results]
    branchy_accs = [r['accuracy'] for r in branchy_results]
    plt.plot(branchy_costs, branchy_accs, 'k-o', linewidth=2, markersize=8, label='BranchyNet')
    
    # EM Routing (one line per lambda)
    colors = plt.cm.viridis([i/len(lambda_vals) for i in range(len(lambda_vals))])
    for i, lam in enumerate(lambda_vals):
        lam_results = [r for r in em_results if r['lambda'] == lam]
        costs = [r['cost'] for r in lam_results]
        accs = [r['accuracy'] for r in lam_results]
        plt.plot(costs, accs, '-s', color=colors[i], linewidth=1.5, markersize=6, label=f'EM λ={lam}')
    
    plt.xlabel('Computational Cost', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('BranchyNet vs EM Routing: Accuracy-Cost Trade-off', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/branchy_vs_em.png', dpi=150)
    print("\nPlot saved to results/branchy_vs_em.png")

if __name__ == "__main__":
    run_comparison()
