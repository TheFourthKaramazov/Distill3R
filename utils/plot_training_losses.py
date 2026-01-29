#!/usr/bin/env python3
"""
Plot training losses from PyTorch Lightning CSV logs.

This script automatically finds all 'version_X/metrics.csv' files
in a given log directory, concatenates them, and plots the continuous
training run.

Usage:
    # Basic usage (linear scale)
    python utils/plot_training_losses.py logs/distill3r/distill3r/

    # With custom output path
    python utils/plot_training_losses.py logs/distill3r/distill3r/ --output results/custom.png

    # With log-scale plot (generates both linear and log)
    python utils/plot_training_losses.py logs/distill3r/distill3r/ --log-scale

    # With efficiency metrics
    python utils/plot_training_losses.py logs/distill3r/distill3r/ --with-efficiency

    # All options combined
    python utils/plot_training_losses.py logs/distill3r/distill3r/ --log-scale --with-efficiency
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np


def load_and_clean_metrics(log_base_path):
    """load and concatenate all 'version_*/metrics.csv' files."""
    base_path = Path(log_base_path)
    
    # find all metrics.csv files and sort them by version number
    csv_files = sorted(
        base_path.glob('version_*/metrics.csv'),
        key=lambda p: int(p.parent.name.split('_')[-1])
    )
    
    if not csv_files:
        raise FileNotFoundError(f"no 'version_*/metrics.csv' files found in {base_path}")

    print(f"found {len(csv_files)} log files to concatenate:")
    for f in csv_files:
        print(f"  - {f}")

    # load all dataframes into a list
    df_list = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"warning: {csv_path} is empty, skipping.")
        except Exception as e:
            print(f"error loading {csv_path}: {e}")

    if not df_list:
        raise ValueError("all found csv files were empty or failed to load.")

    # concatenate all dataframes
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"\ntotal rows (all versions): {len(full_df)}")

    # sort by step and drop duplicates, keeping the last entry
    # this handles any overlaps from resumed runs correctly
    full_df = full_df.sort_values(by='step')
    full_df = full_df.drop_duplicates(subset='step', keep='last')
    print(f"total rows (after deduplication): {len(full_df)}")

    # filter rows that have step loss values (not nan)
    # we use train/total_step as indicator since it's always present
    df_losses = full_df[full_df['train/total_step'].notna()].copy()
    print(f"rows with loss data: {len(df_losses)}")

    if df_losses.empty:
        print("warning: no rows with loss data found after filtering.")
        return df_losses

    # extract step number for x-axis
    df_losses['global_step'] = df_losses['step']

    return df_losses


def plot_losses(df, output_path=None, title_suffix="", log_scale=False):
    """Plot all training losses.

    Args:
        df: DataFrame with loss data
        output_path: Path to save plot
        title_suffix: Additional text for title
        log_scale: If True, use log scale for y-axis
    """

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    scale_label = " (Log Scale)" if log_scale else ""
    fig.suptitle(f'Training Losses{title_suffix}{scale_label}', fontsize=16, fontweight='bold')

    # Loss types to plot
    losses = [
        ('train/global_step', 'Global Loss (L2 Point-map)', 'tab:blue'),
        ('train/local_step', 'Local Loss (L2 Coordinates)', 'tab:orange'),
        ('train/confidence_step', 'Confidence Loss (L1)', 'tab:green'),
        ('train/total_step', 'Total Loss (Weighted Sum)', 'tab:red')
    ]

    for idx, (loss_col, title, color) in enumerate(losses):
        ax = axes[idx // 2, idx % 2]
        
        if loss_col not in df.columns:
            print(f"warning: column '{loss_col}' not found. skipping plot.")
            ax.set_title(f"{title}\n(data not found)", fontsize=12, fontweight='bold', color='red')
            continue

        # Get data
        x = df['global_step'].values
        y = df[loss_col].values
        
        if len(x) == 0:
            print(f"warning: no data for '{loss_col}'. skipping plot.")
            ax.set_title(f"{title}\n(no data)", fontsize=12, fontweight='bold', color='red')
            continue

        # Plot
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.7)

        # Add smoothed line (moving average over 50 points)
        if len(y) > 50:
            window = min(50, len(y) // 20)  # Adaptive window size
            if window < 1: window = 1
            y_smooth = pd.Series(y).rolling(window=window, center=True).mean()
            ax.plot(x, y_smooth, color=color, linewidth=2.5, label=f'MA({window})')
            ax.legend()

        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')

        # Formatting
        ax.set_xlabel('Global Step', fontsize=11)
        ylabel = 'Loss (log scale)' if log_scale else 'Loss'
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        mean_val = np.mean(y)
        final_val = y[-1]
        min_val = np.min(y)
        if log_scale:
            # Use scientific notation for log scale
            textstr = f'Mean: {mean_val:.2e}\nFinal: {final_val:.2e}\nMin: {min_val:.2e}'
        else:
            textstr = f'Mean: {mean_val:.4f}\nFinal: {final_val:.4f}\nMin: {min_val:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def print_loss_summary(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("LOSS SUMMARY")
    print("="*70)

    losses = [
        ('train/global_step', 'Global Loss'),
        ('train/local_step', 'Local Loss'),
        ('train/confidence_step', 'Confidence Loss'),
        ('train/total_step', 'Total Loss')
    ]

    for loss_col, name in losses:
        if loss_col not in df.columns:
            print(f"\n{name}: data not found.")
            continue
            
        values = df[loss_col].values
        if len(values) == 0:
            print(f"\n{name}: no data.")
            continue

        print(f"\n{name}:")
        print(f"  Initial: {values[0]:.6f}")
        print(f"  Final:   {values[-1]:.6f}")
        print(f"  Min:     {np.min(values):.6f}")
        print(f"  Mean:    {np.mean(values):.6f}")
        print(f"  Std:     {np.std(values):.6f}")
        if values[0] != 0:
            print(f"  Improvement: {((values[0] - values[-1]) / values[0] * 100):.2f}%")
        else:
            print("  Improvement: N/A (initial value is 0)")

    # Training info
    epochs = df['epoch'].max()
    steps = df['global_step'].max()
    print(f"\n{'='*70}")
    print(f"Training Progress: {epochs:.0f} epochs, {steps:.0f} steps")
    print(f"Total data points: {len(df)}")
    print("="*70 + "\n")


def plot_efficiency(memory_csv_path, output_path=None):
    """Plot training efficiency metrics."""
    print(f"\nLoading efficiency data from {memory_csv_path}...")
    
    try:
        df = pd.read_csv(memory_csv_path)
    except FileNotFoundError:
        print(f"error: efficiency file not found at {memory_csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"error: efficiency file {memory_csv_path} is empty.")
        return
        
    print(f"Memory log rows: {len(df)}")
    if len(df) < 2:
        print("error: not enough data to plot efficiency.")
        return

    # Calculate throughput
    df['steps_per_sec'] = df['step'].diff() / df['elapsed_sec'].diff()
    df['samples_per_sec'] = df['steps_per_sec'] * 4  # batch_size=4

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Efficiency Metrics', fontsize=16, fontweight='bold')

    # Throughput
    ax = axes[0, 0]
    x = df['step'].values[1:]
    y = df['steps_per_sec'].values[1:]
    ax.plot(x, y, color='tab:purple', linewidth=1.0, alpha=0.7)
    if len(y) > 50:
        y_smooth = pd.Series(y).rolling(window=50, center=True).mean()
        ax.plot(x, y_smooth, color='tab:purple', linewidth=2.5, label='MA(50)')
        ax.legend()
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Steps/sec', fontsize=11)
    ax.set_title('Training Throughput', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    mean_throughput = np.nanmean(y)
    ax.text(0.02, 0.98, f'Mean: {mean_throughput:.3f} steps/sec',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # GPU Memory
    ax = axes[0, 1]
    x = df['step'].values
    y = df['gpu_memory_gb'].values
    ax.plot(x, y, color='tab:red', linewidth=1.5)
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('GPU Memory (GB)', fontsize=11)
    ax.set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, f'Mean: {np.mean(y):.2f} GB',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # System Memory
    ax = axes[1, 0]
    x = df['step'].values
    y = df['system_memory_gb'].values
    ax.plot(x, y, color='tab:orange', linewidth=1.5)
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('System Memory (GB)', fontsize=11)
    ax.set_title('System Memory Usage', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, f'Mean: {np.mean(y):.2f} GB',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Time per epoch estimate
    ax = axes[1, 1]
    steps_per_epoch = 3501  # 28009 samples / 4 batch / 2 accum
    hours = df['elapsed_sec'].values / 3600
    epochs = df['step'].values / steps_per_epoch
    ax.plot(epochs, hours, color='tab:green', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Total Time (hours)', fontsize=11)
    ax.set_title('Cumulative Training Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Estimate total time
    if len(epochs) > 10:
        # Linear fit for time projection
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(epochs[epochs > 0], hours[epochs > 0], 1)
        time_per_epoch = p.coef[1]  # hours per epoch
        projected_100 = time_per_epoch * 100
        projected_200 = time_per_epoch * 200
        textstr = f'Time/epoch: {time_per_epoch:.2f}h\n100 epochs: {projected_100:.1f}h ({projected_100/24:.1f}d)\n200 epochs: {projected_200:.1f}h ({projected_200/24:.1f}d)'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved efficiency plot to {output_path}")
    else:
        plt.show()

    plt.close()

    # Print efficiency summary
    print("\n" + "="*70)
    print("EFFICIENCY SUMMARY")
    print("="*70)
    throughput = df['steps_per_sec'].values[1:]
    print(f"Throughput: {np.nanmean(throughput):.3f} steps/sec (mean)")
    print(f"            {np.nanmean(throughput) * 4:.2f} samples/sec")
    print(f"GPU Memory: {df['gpu_memory_gb'].mean():.2f} GB (mean)")
    print(f"System RAM: {df['system_memory_gb'].mean():.2f} GB (mean)")
    print(f"Total time: {df['elapsed_sec'].iloc[-1] / 3600:.2f} hours")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training losses from Lightning CSV logs (concatenates all versions)')
    
    # --- modified argument ---
    parser.add_argument('log_dir', type=str, 
                        help='Path to the base log directory containing version_X folders (e.g., logs/full_training/distill3r_full/)')
    # -------------------------

    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for plot (default: results/full_training/stats/losses_epochN.png)')
    parser.add_argument('--with-efficiency', action='store_true',
                        help='Also plot efficiency metrics from memory_usage.csv')
    parser.add_argument('--log-scale', action='store_true',
                        help='Generate both linear and log-scale plots')
    args = parser.parse_args()

    # Load data
    try:
        df = load_and_clean_metrics(args.log_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}")
        return
        
    if df.empty:
        print("error: no loss data found. exiting.")
        return

    # Print summary
    print_loss_summary(df)

    # Get epoch info for title
    current_epoch = int(df['epoch'].max())
    title_suffix = f" (Epoch {current_epoch})"

    # Determine output path
    output_path = args.output
    if output_path is None:
        # Default: results/full_training/stats/losses_epochN.png
        output_path = f"results/distill3r_shuffle/stats/losses_epoch{current_epoch}.png"

    # Plot losses (linear scale)
    plot_losses(df, output_path=output_path, title_suffix=title_suffix, log_scale=False)

    # Plot losses (log scale) if requested
    if args.log_scale:
        log_output = str(Path(output_path).with_name(Path(output_path).stem + "_log.png"))
        plot_losses(df, output_path=log_output, title_suffix=title_suffix, log_scale=True)

    # Plot efficiency if requested
    # --- LIGNE CORRIGÉE ---
    if args.with_efficiency:
        # --- modified path logic ---
        # look for memory_usage.csv in the parent of the log_dir
        # (e.g., logs/full_training/distill3r_full/ -> parent is logs/full_training/)
        log_dir_path = Path(args.log_dir)
        memory_csv = log_dir_path.parent / "memory_usage.csv"
        # ---------------------------

        if memory_csv.exists():
            efficiency_output = str(Path(output_path).parent / f"efficiency_epoch{current_epoch}.png")
            plot_efficiency(str(memory_csv), output_path=efficiency_output)
        else:
            print(f"warning: memory_usage.csv not found at {memory_csv}")

if __name__ == '__main__':
    main()