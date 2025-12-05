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

    # Data is organized by Lambda
    groups = {}
    for entry in data:
        lam = entry['lambda']
        if lam not in groups:
            groups[lam] = {'cost': [], 'acc': [], 'thresh': []}
        groups[lam]['cost'].append(entry['cost'])
        groups[lam]['acc'].append(entry['accuracy'])
        groups[lam]['thresh'].append(entry['threshold'])

    plt.figure(figsize=(10, 6))
    
    # Plot each lambda curve
    for lam, values in groups.items():
        # Sort by cost to make a nice line
        sorted_indices = sorted(range(len(values['cost'])), key=lambda k: values['cost'][k])
        costs = [values['cost'][i] for i in sorted_indices]
        accs = [values['acc'][i] for i in sorted_indices]
        
        plt.plot(costs, accs, marker='o', label=f'Lambda={lam}')

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
