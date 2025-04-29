import os
import sys
import numpy as np
import pandas as pd
from glob import glob

# Path to output directory (adjust if needed)
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def summarize():
    # Find all evaluation reward files
    eval_pattern = os.path.join(BASE_DIR, "*_eval_rewards.txt")
    eval_files = glob(eval_pattern)
    if not eval_files:
        print(f"No eval reward files found in {BASE_DIR}")
        sys.exit(1)

    records = []
    for eval_path in eval_files:
        # Param key is filename without suffix
        fname = os.path.basename(eval_path)
        params = fname.replace("_eval_rewards.txt", "")

        # Load eval rewards
        try:
            eval_rewards = np.loadtxt(eval_path)
        except Exception as e:
            print(f"Failed to load {eval_path}: {e}")
            continue
        n = len(eval_rewards)
        if n == 0:
            print(f"No data in {eval_path}")
            continue

        # Compute win rate: >0 => win; ==0 => draw; else loss
        wins = np.sum(np.where(eval_rewards > 0, 1.0,
                                np.where(eval_rewards == 0, 0.5, 0.0)))
        win_rate = wins / n
        avg_eval = eval_rewards.mean()

        # Load training episode rewards if present
        train_path = os.path.join(BASE_DIR, f"{params}_episode_rewards.txt")
        if os.path.exists(train_path):
            try:
                train_rewards = np.loadtxt(train_path)
                avg_train = train_rewards.mean() if len(train_rewards)>0 else np.nan
            except:
                avg_train = np.nan
        else:
            avg_train = np.nan

        records.append({
            'params': params,
            'win_rate': win_rate,
            'avg_eval_reward': avg_eval,
            'avg_train_reward': avg_train,
            'n_eval': n
        })

    if not records:
        print("No valid eval records to summarize.")
        sys.exit(1)

    df = pd.DataFrame(records)
    # Sort by win_rate
    df = df.sort_values(by='win_rate', ascending=False).reset_index(drop=True)
    return df


if __name__ == '__main__':
    summary_df = summarize()
    print("\n=== Hyperparam Sweep Summary ===")
    print(summary_df.to_string(index=False))

    # Save to CSV
    out_csv = os.path.join(BASE_DIR, 'summary_results.csv')
    try:
        summary_df.to_csv(out_csv, index=False)
        print(f"\nSaved summary to {out_csv}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
