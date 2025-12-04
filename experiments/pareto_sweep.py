import torch
import json
import os
from main import train_models, evaluate_models, Config

def run_pareto_sweep():
    # Define Sweep Params
    lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    os.makedirs("results", exist_ok=True)
    
    #Train Routers with different Lambdas
    for lam in lambda_vals:
        print(f"\n--- Training with Lambda={lam} ---")
        # Train routers for this specific lambda
        train_models("train_routers", lambda_val=lam)
        # Evaluate with different Thresholds
        for thresh in thresholds:
            print(f"   Evaluating Threshold={thresh:.2f}...")
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
            with open("results/pareto_results.json", "w") as f:
                json.dump(results, f, indent=4)
                
    print("\n=== Pareto Sweep Complete ===")
    print(f"Results saved to results/pareto_results.json")

if __name__ == "__main__":
    run_pareto_sweep()
