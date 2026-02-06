#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
import re
import subprocess
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Run FEMNIST comparison experiments")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with few rounds"
    )
    parser.add_argument(
        "--alg",
        nargs="+",
        default=["fedproto"],
        help="List of algorithms to run (default: fedproto)",
    )
    return parser.parse_args()


def extract_accuracy(output: str) -> Dict[str, float]:
    results = {}

    # 1. Parse final results (Final Round)
    with_protos_match = re.search(
        r"For all users \(with protos\), mean of test acc is ([\d\.]+)", output
    )
    if with_protos_match:
        results["final_with_protos"] = float(with_protos_match.group(1))

    without_protos_match = re.search(
        r"For all users \(w/o protos\), mean of test acc is ([\d\.]+)", output
    )
    if without_protos_match:
        results["final_without_protos"] = float(without_protos_match.group(1))

    # 2. Parse all rounds to find Max Accuracy (Best Score)
    # Pattern: "Round 120, Global Acc: 0.85000, Local Acc: 0.84000, Proto Loss: 0.12345"
    # Global Acc = with protos, Local Acc = without protos
    round_pattern = re.compile(
        r"Round\s+(\d+),\s+Global Acc:\s+([\d\.]+),\s+Local Acc:\s+([\d\.]+)"
    )

    max_global = 0.0
    max_local = 0.0

    for match in round_pattern.finditer(output):
        g_acc = float(match.group(2))
        l_acc = float(match.group(3))
        if g_acc > max_global:
            max_global = g_acc
        if l_acc > max_local:
            max_local = l_acc

    results["max_with_protos"] = max_global
    results["max_without_protos"] = max_local

    return results


def run_experiment(params: Dict[str, any], rounds: int) -> Dict[str, any]:
    print(f"Running experiment with params: {params}")

    # Fixed hyperparameters from paper
    cmd = [
        "uv",
        "run",
        "experiments/federated_main.py",
        "--dataset",
        "femnist",
        "--model",
        "cnn",
        "--rounds",
        str(rounds),
        "--local_bs",
        "8",
        "--lr",
        "0.01",
        "--momentum",
        "0.5",
        "--ld",
        "1",
        "--train_ep",
        "1",
        "--num_classes",
        "62",
        "--shots",
        "100",
        "--gpu",
        "0",
    ]

    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    try:
        # Run command and capture output
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = result.stdout

        # Parse results
        accuracies = extract_accuracy(output)
        return {**params, **accuracies, "status": "Success"}

    except subprocess.CalledProcessError as e:
        print(f"Experiment failed for params {params}")
        print("Error output:", e.stderr)
        return {**params, "status": "Failed", "error": str(e)}


def print_table(results: List[Dict[str, any]]):
    if not results:
        print("No results to display.")
        return

    # Determine columns dynamically
    all_keys = set().union(*(d.keys() for d in results))
    # Order: input params, then results
    param_keys = ["alg", "ways", "stdev", "seed"]
    # Show Final and Max for "with protos" (Global) and "without protos" (Local)
    metric_keys = [
        "final_with_protos",
        "max_with_protos",
        "final_without_protos",
        "max_without_protos",
        "status",
    ]

    headers = [k for k in param_keys if k in all_keys] + [
        k for k in metric_keys if k in all_keys
    ]

    header_str = " | ".join(f"{h:<20}" for h in headers)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    for row in results:
        values = []
        for h in headers:
            val = row.get(h, "N/A")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        print(" | ".join(f"{v:<20}" for v in values))


def main():
    args = parse_args()

    # Define parameter grid
    ways_list = [3, 4, 5]
    stdev_list = [1, 2]
    seed_list = [1234, 5678, 9012]

    # If in test mode, override for speed
    rounds = 120
    if args.test:
        rounds = 2
        # Reduce grid for test
        ways_list = [3]
        stdev_list = [1]
        seed_list = [1234]
        # Overwrite alg list if user didn't specify multiple, or just keep what they passed?
        # Respect user input for alg even in test mode, unless we want to force simple.
        # Let's keep args.alg as is.
        print("Test mode enabled: running with reduced parameters and rounds.")

    results = []

    # Iterate over algs as well
    alg_list = args.alg

    for alg, ways, stdev, seed in itertools.product(
        alg_list, ways_list, stdev_list, seed_list
    ):
        params = {"alg": alg, "ways": ways, "stdev": stdev, "seed": seed}
        res = run_experiment(params, rounds)
        results.append(res)

    print("\n\nExperiment Summary:")
    print_table(results)


if __name__ == "__main__":
    main()
