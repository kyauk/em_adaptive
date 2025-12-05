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
    # Sort by cost
    all_points.sort(key=lambda x: x[0])
    
    pareto_front = []
    current_max_acc = -1.0
    
    # We want the max accuracy for any cost <= c. 
    # Actually, standard Pareto is: for a given cost, maximize accuracy.
    # So we iterate through sorted costs. If a point has higher accuracy than any previous point, it's on the frontier.
    # BUT, we might have a point with higher cost but LOWER accuracy (dominated).
    # We want the "upper left" boundary.
    
    # Simple algorithm:
    # 1. Sort by Cost
    # 2. Iterate and keep point if Acc > max_acc_so_far
    # Wait, that's for "min cost for max acc".
    # We want "max acc for min cost".
    
    # Let's do: Sort by Accuracy DESCENDING.
    # Then keep point if Cost < min_cost_so_far.
    
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
    
    plt.plot(p_costs, p_accs, 'k--', linewidth=2, label='Pareto Frontier')

    plt.xlabel('Computational Cost (Normalized)')
    plt.ylabel('Accuracy')
    plt.title('EM Routing: Accuracy vs Cost (Pareto Frontier)')
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
