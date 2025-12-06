
import torch
import json
import os
import sys
from main import train_models, evaluate_models

def run_lambda_curves_sweep():
    # Sweep Params
    lambda_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = []
    
    os.makedirs("results", exist_ok=True)
    results_file = "results/lambda_curves_results.json"
    
    # Train Routers with different Lambdas
    for lam in lambda_vals:
        print(f"\nTraining with Lambda={lam}...")
        # Train routers for this specific lambda
        train_models("train_routers", lambda_val=lam)
        
        # Evaluate with different Thresholds
        for thresh in thresholds:
            print(f"Evaluating Threshold={thresh:.2f}...")
            metrics = evaluate_models("em_routing", threshold=thresh)
            
            # store results
            entry = {
                "lambda": lam,
                "threshold": thresh,
                "accuracy": float(metrics["accuracy"]),
                "cost": float(metrics["cost"])
            }
            results.append(entry)
            
            # Save incrementally
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
                
    print("\nLAMBDA CURVES SWEEP COMPLETE\n")
    print(f"Results saved to {results_file}")
    
    # Generate the Plot
    print("\nGenerating Lambda Curves Plot...")
    from experiments.visualization import plot_lambda_curves
    plot_lambda_curves()

if __name__ == "__main__":
    run_lambda_curves_sweep()
