import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cloud servers
import matplotlib.pyplot as plt
import os

def plot_pareto_frontier():
    results_path = "results/pareto_results.json"
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run pareto_sweep.py first.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Collect all points
    all_points = []
    for entry in data:
        all_points.append((entry['cost'], entry['accuracy']))

    # Data is organized by Lambda for scatter plot
    groups = {}
    for entry in data:
        lam = entry['lambda']
        if lam not in groups:
            groups[lam] = {'cost': [], 'acc': [], 'thresh': []}
        groups[lam]['cost'].append(entry['cost'])
        groups[lam]['acc'].append(entry['accuracy'])
        groups[lam]['thresh'].append(entry['threshold'])

    plt.figure(figsize=(10, 6))
    
    # Plot each lambda curve as scatter
    for lam, values in groups.items():
        plt.scatter(values['cost'], values['acc'], label=f'Lambda={lam}', alpha=0.6)

    # Compute Pareto Frontier (Convex Hull / Upper Envelope)
    # We want to find the set of points that maximize accuracy for a given cost.
    # Algorithm: Sort by Accuracy (descending), then keep point if Cost < min_cost_so_far.
    
    all_points_by_acc = sorted(all_points, key=lambda x: x[1], reverse=True)
    pareto_points = []
    min_cost = float('inf')
    
    for cost, acc in all_points_by_acc:
        if cost < min_cost:
            pareto_points.append((cost, acc))
            min_cost = cost
            
    # Sort back by cost for plotting
    pareto_points.sort(key=lambda x: x[0])
    
    # Extract x and y
    p_costs = [p[0] for p in pareto_points]
    p_accs = [p[1] for p in pareto_points]
    
    plt.plot(p_costs, p_accs, 'k--', linewidth=2, label='EM Pareto Frontier')

    # --- Add BranchyNet Baseline ---
    # We run a quick evaluation here or hardcode points if we don't want to import everything.
    # Better to import and run since it's fast (no training).
    try:
        from main import setup_model, get_dataloader
        from experiments.evaluation import Evaluator
        
        print("Running BranchyNet baseline for comparison...")
        model = setup_model(load_exits=True)
        evaluator = Evaluator(model)
        _, test_loader = get_dataloader()
        
        branchy_thresholds = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5]
        b_costs = []
        b_accs = []
        
        for t in branchy_thresholds:
            res = evaluator.eval_branchynet(test_loader, threshold=t)
            b_costs.append(res['cost'])
            b_accs.append(res['accuracy'])
            
        # Sort for plotting
        b_points = sorted(zip(b_costs, b_accs), key=lambda x: x[0])
        b_costs = [p[0] for p in b_points]
        b_accs = [p[1] for p in b_points]
        
        plt.plot(b_costs, b_accs, 'r-^', linewidth=2, label='BranchyNet')
        
    except Exception as e:
        print(f"Could not run BranchyNet baseline: {e}")

    plt.xlabel('Computational Cost (Normalized)')
    plt.ylabel('Accuracy')
    plt.title('EM Routing vs BranchyNet: Pareto Frontier')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = "results/pareto_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def plot_comparison_bars(results):
    """
    Plot bar chart comparing all methods.
    results: dict returned by eval_all
    """
    methods = []
    accs = []
    costs = []
    
    # Filter and order methods
    priority = ['resnet', 'branchynet', 'em_routing', 'oracle', 'random']
    for m in priority:
        if m in results:
            methods.append(m)
            accs.append(results[m]['accuracy'])
            costs.append(results[m]['cost'])
            
    # Add fixed exits if present
    for k in sorted(results.keys()):
        if k.startswith('fixed_'):
            methods.append(k)
            accs.append(results[k]['accuracy'])
            costs.append(results[k]['cost'])

    x = range(len(methods))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Accuracy Bars
    bars1 = ax1.bar([i - width/2 for i in x], accs, width, label='Accuracy', color='skyblue')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    
    # Cost Bars (on secondary axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], costs, width, label='Cost', color='salmon')
    ax2.set_ylabel('Normalized Cost')
    ax2.set_ylim(0, 1.05)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Method Comparison: Accuracy vs Cost')
    plt.tight_layout()
    
    output_path = "results/comparison_bar_chart.png"
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    plot_pareto_frontier()
