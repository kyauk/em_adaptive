import json
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

if __name__ == "__main__":
    plot_pareto_frontier()
