"""
Visualization utilities for dataset conversion debugging.

These functions help verify that data was loaded and converted correctly.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import json


def plot_folder_structure(base_path: Path, max_depth: int = 3):
    """
    Visualize the folder structure as a tree.

    Args:
        base_path: Root directory to visualize
        max_depth: Maximum depth to traverse
    """
    def tree_structure(directory: Path, prefix: str = "", depth: int = 0):
        """Recursively build tree structure."""
        if depth > max_depth:
            return []

        lines = []
        try:
            contents = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            for idx, path in enumerate(contents):
                is_last = idx == len(contents) - 1
                current_prefix = "└── " if is_last else "├── "

                if path.is_dir():
                    # Count items in directory
                    try:
                        item_count = len(list(path.iterdir()))
                        lines.append(f"{prefix}{current_prefix}{path.name}/ ({item_count} items)")
                    except:
                        lines.append(f"{prefix}{current_prefix}{path.name}/")

                    # Recurse for subdirectories
                    if depth < max_depth:
                        extension = "    " if is_last else "│   "
                        lines.extend(tree_structure(path, prefix + extension, depth + 1))
                else:
                    # Show file with size
                    size = path.stat().st_size
                    size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                    lines.append(f"{prefix}{current_prefix}{path.name} ({size_str})")
        except PermissionError:
            pass

        return lines

    print("\n" + "=" * 80)
    print(f"Folder Structure: {base_path}")
    print("=" * 80)
    print(base_path.name + "/")

    lines = tree_structure(base_path)
    for line in lines[:100]:  # Limit to first 100 lines
        print(line)

    if len(lines) > 100:
        print(f"... ({len(lines) - 100} more lines)")


def plot_sample_sessions(dataset_path: Path, num_samples: int = 3):
    """
    Plot time series from sample sessions.

    Args:
        dataset_path: Path to dataset (e.g., data/uci_har)
        num_samples: Number of random sessions to plot
    """
    sessions_dir = dataset_path / "sessions"
    session_dirs = list(sessions_dir.glob("*/"))

    if not session_dirs:
        print("No sessions found!")
        return

    # Sample random sessions
    num_samples = min(num_samples, len(session_dirs))
    sampled = np.random.choice(session_dirs, num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx, session_dir in enumerate(sampled):
        parquet_path = session_dir / "data.parquet"
        df = pd.read_parquet(parquet_path)

        # Plot first few channels
        ax = axes[idx]
        channels = [col for col in df.columns if col != 'timestamp_sec'][:5]

        for ch in channels:
            ax.plot(df['timestamp_sec'], df[ch], label=ch, alpha=0.7)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Value')
        ax.set_title(f'Session: {session_dir.name} ({len(df)} samples)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = dataset_path / "debug_sample_sessions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved sample session plots: {output_path}")
    plt.close()


def plot_channel_distributions(dataset_path: Path, max_channels: int = 12):
    """
    Plot distribution histograms for channels.

    Args:
        dataset_path: Path to dataset
        max_channels: Maximum number of channels to plot
    """
    # Load manifest to get channel names
    manifest_path = dataset_path / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    channel_names = [ch['name'] for ch in manifest['channels']][:max_channels]

    # Collect data from a subset of sessions
    sessions_dir = dataset_path / "sessions"
    session_dirs = list(sessions_dir.glob("*/"))[:20]  # Sample first 20 sessions

    channel_data = {ch: [] for ch in channel_names}

    print(f"\nCollecting data from {len(session_dirs)} sessions for distribution plots...")
    for session_dir in session_dirs:
        parquet_path = session_dir / "data.parquet"
        df = pd.read_parquet(parquet_path)

        for ch in channel_names:
            if ch in df.columns:
                # Only add numeric data
                if pd.api.types.is_numeric_dtype(df[ch]):
                    channel_data[ch].extend(df[ch].dropna().values[:1000])  # Sample 1000 points

    # Plot distributions
    n_channels = len([ch for ch in channel_names if channel_data[ch]])

    if n_channels == 0:
        print("  No channel data to plot (no sessions or no numeric channels)")
        return

    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_channels > 1 else [axes]

    plot_idx = 0
    for ch in channel_names:
        if not channel_data[ch]:
            continue

        ax = axes[plot_idx]
        data = np.array(channel_data[ch])

        # Skip if data is not numeric
        if not np.issubdtype(data.dtype, np.number):
            print(f"  Warning: Skipping non-numeric channel '{ch}' (dtype: {data.dtype})")
            continue

        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{ch}\nμ={np.mean(data):.2f}, σ={np.std(data):.2f}', fontsize=9)
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = dataset_path / "debug_channel_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved channel distribution plots: {output_path}")
    plt.close()


def plot_session_duration_histogram(dataset_path: Path):
    """
    Plot histogram of session durations.

    Args:
        dataset_path: Path to dataset
    """
    sessions_dir = dataset_path / "sessions"
    session_dirs = list(sessions_dir.glob("*/"))

    if not session_dirs:
        print("  No sessions found - skipping duration histogram")
        return

    durations = []
    for session_dir in session_dirs:
        parquet_path = session_dir / "data.parquet"
        df = pd.read_parquet(parquet_path)
        duration = df['timestamp_sec'].iloc[-1] - df['timestamp_sec'].iloc[0]
        durations.append(duration)

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Sessions')
    plt.title(f'Session Duration Distribution\n'
              f'Total: {len(durations)} sessions, '
              f'Mean: {np.mean(durations):.1f}s, '
              f'Median: {np.median(durations):.1f}s')
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Min: {np.min(durations):.1f}s\n' \
                 f'Max: {np.max(durations):.1f}s\n' \
                 f'Total: {np.sum(durations)/3600:.2f}h'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    output_path = dataset_path / "debug_session_durations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved session duration histogram: {output_path}")
    plt.close()


def plot_label_distribution(dataset_path: Path):
    """
    Plot distribution of activity labels.

    Args:
        dataset_path: Path to dataset
    """
    labels_path = dataset_path / "labels.json"
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    if not labels:
        print("  No labels found - skipping label distribution")
        return

    # Count label occurrences
    label_counts = {}
    for session_labels in labels.values():
        for label in session_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Sort by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    # Plot
    plt.figure(figsize=(12, 6))
    labels_list, counts = zip(*sorted_labels)

    plt.bar(range(len(labels_list)), counts, alpha=0.7, edgecolor='black')
    plt.xlabel('Activity')
    plt.ylabel('Number of Sessions')
    plt.title(f'Activity Label Distribution ({len(labels)} total sessions)')
    plt.xticks(range(len(labels_list)), labels_list, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = dataset_path / "debug_label_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved label distribution plot: {output_path}")
    plt.close()


def generate_debug_visualizations(dataset_path: Path):
    """
    Generate all debug visualizations for a converted dataset.

    Args:
        dataset_path: Path to dataset directory
    """
    print("\n" + "=" * 80)
    print("Generating Debug Visualizations")
    print("=" * 80)

    # 1. Folder structure
    plot_folder_structure(dataset_path, max_depth=2)

    # 2. Sample sessions
    plot_sample_sessions(dataset_path, num_samples=3)

    # 3. Channel distributions
    plot_channel_distributions(dataset_path, max_channels=12)

    # 4. Session durations
    plot_session_duration_histogram(dataset_path)

    # 5. Label distribution
    plot_label_distribution(dataset_path)

    print("\n" + "=" * 80)
    print("Debug visualizations complete!")
    print(f"Check {dataset_path}/ for debug_*.png files")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization_utils.py <dataset_path>")
        print("Example: python visualization_utils.py data/capture24")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    generate_debug_visualizations(dataset_path)
