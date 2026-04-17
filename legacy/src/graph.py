import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

# Load the data
runtime_logs_path = Path(__file__).parent.parent / "runtime_logs.csv"
run_summary_path = Path(__file__).parent.parent / "run_summary.csv"

runtime_df = pd.read_csv(runtime_logs_path)
summary_df = pd.read_csv(run_summary_path)

# Filter out summary rows
data_df = runtime_df[~runtime_df['sample_id'].str.startswith('SUMMARY')]

# Extract run configurations
runs = {
    '1_model0': 'Baseline (No TensorRT)',
    '2_model0': 'TensorRT FP32',
    '3_model0_fp16': 'TensorRT FP16'
}

# Create output directory for graphs
output_dir = Path(__file__).parent.parent / "graphs"
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("INFERENCE RUNTIME COMPARISON ANALYSIS")
print("=" * 80)

# 1. Extract inference times for each run
inference_times = {}
load_times = {}
write_times = {}
total_times = {}

for run_id, run_name in runs.items():
    inference_times[run_name] = data_df[f"{run_id}_inference_time"].values
    load_times[run_name] = data_df[f"{run_id}_bsq_load_time"].values
    write_times[run_name] = data_df[f"{run_id}_write_time"].values
    total_times[run_name] = data_df[f"{run_id}_total_time"].values

# 2. Calculate speedup statistics
baseline = inference_times['Baseline (No TensorRT)']
fp32_speedup = baseline / inference_times['TensorRT FP32']
fp16_speedup = baseline / inference_times['TensorRT FP16']

print(f"\n{'SPEEDUP STATISTICS':-^80}")
print(f"\n{'Metric':<30} {'TensorRT FP32':<20} {'TensorRT FP16':<20}")
print("-" * 70)
print(f"{'Mean Speedup:':<30} {fp32_speedup.mean():.3f}x{'':<14} {fp16_speedup.mean():.3f}x")
print(f"{'Median Speedup:':<30} {np.median(fp32_speedup):.3f}x{'':<14} {np.median(fp16_speedup):.3f}x")
print(f"{'Min Speedup:':<30} {fp32_speedup.min():.3f}x{'':<14} {fp16_speedup.min():.3f}x")
print(f"{'Max Speedup:':<30} {fp32_speedup.max():.3f}x{'':<14} {fp16_speedup.max():.3f}x")
print(f"{'Std Dev:':<30} {fp32_speedup.std():.3f}{'':<15} {fp16_speedup.std():.3f}")

# Print detailed statistics for all time categories
print(f"\n{'INFERENCE TIME STATISTICS (seconds)':-^80}")
print(f"\n{'Run':<30} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12}")
print("-" * 90)
for run_name, times in inference_times.items():
    print(f"{run_name:<30} {times.mean():<12.4f} {np.median(times):<12.4f} {times.min():<12.4f} {times.max():<12.4f} {times.std():<12.4f}")

print(f"\n{'LOAD TIME STATISTICS (seconds)':-^80}")
print(f"\n{'Run':<30} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12}")
print("-" * 90)
for run_name, times in load_times.items():
    print(f"{run_name:<30} {times.mean():<12.4f} {np.median(times):<12.4f} {times.min():<12.4f} {times.max():<12.4f} {times.std():<12.4f}")

print(f"\n{'WRITE TIME STATISTICS (seconds)':-^80}")
print(f"\n{'Run':<30} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12}")
print("-" * 90)
for run_name, times in write_times.items():
    print(f"{run_name:<30} {times.mean():<12.4f} {np.median(times):<12.4f} {times.min():<12.4f} {times.max():<12.4f} {times.std():<12.4f}")

print(f"\n{'TOTAL TIME STATISTICS (seconds)':-^80}")
print(f"\n{'Run':<30} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12}")
print("-" * 90)
for run_name, times in total_times.items():
    print(f"{run_name:<30} {times.mean():<12.4f} {np.median(times):<12.4f} {times.min():<12.4f} {times.max():<12.4f} {times.std():<12.4f}")

# Box plot comparison for inference times
plt.figure(figsize=(12, 8))
data_for_box = [baseline, inference_times['TensorRT FP32'], inference_times['TensorRT FP16']]
labels = ['Baseline\n(No TensorRT)', 'TensorRT\nFP32', 'TensorRT\nFP16']
colors = ['#d62728', '#2ca02c', '#1f77b4']
bp = plt.boxplot(data_for_box, tick_labels=labels, patch_artist=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add text annotations for key statistics
for i, data in enumerate(data_for_box, 1):
    median = np.median(data)
    mean = data.mean()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    # Median (center, with background box)
    plt.text(i, median, f'{median:.3f}', ha='center', va='center', fontsize=10, 
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
             edgecolor='black', alpha=0.8))
    
    # Mean (slightly offset to avoid overlap)
    plt.text(i - 0.15, mean, f'μ: {mean:.3f}', ha='right', va='center', 
             fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor='lightyellow', edgecolor='gray', alpha=0.7))
    
    # Q3 (upper quartile)
    plt.text(i + 0.4, q3, f'Q3: {q3:.3f}', ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
             edgecolor='gray', alpha=0.7))
    
    # Q1 (lower quartile)
    plt.text(i + 0.4, q1, f'Q1: {q1:.3f}', ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
             edgecolor='gray', alpha=0.7))

plt.ylabel('Inference Time (s)', fontsize=12)
plt.title('Inference Time Distribution Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "inference_time_boxplot.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'inference_time_boxplot.png'}")

# Box plot for load times
plt.figure(figsize=(12, 8))
load_data_for_box = [load_times['Baseline (No TensorRT)'], load_times['TensorRT FP32'], load_times['TensorRT FP16']]
bp = plt.boxplot(load_data_for_box, tick_labels=labels, patch_artist=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add text annotations
for i, data in enumerate(load_data_for_box, 1):
    median = np.median(data)
    mean = data.mean()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    plt.text(i, median, f'{median:.3f}', ha='center', va='center', fontsize=10, 
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
             edgecolor='black', alpha=0.8))
    
    plt.text(i - 0.15, mean, f'μ: {mean:.3f}', ha='right', va='center', 
             fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor='lightyellow', edgecolor='gray', alpha=0.7))
    
    plt.text(i + 0.4, q3, f'Q3: {q3:.3f}', ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
             edgecolor='gray', alpha=0.7))
    
    plt.text(i + 0.4, q1, f'Q1: {q1:.3f}', ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
             edgecolor='gray', alpha=0.7))

plt.ylabel('Load Time (s)', fontsize=12)
plt.title('BSQ Load Time Distribution Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "load_time_boxplot.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'load_time_boxplot.png'}")

# Box plot for write times
plt.figure(figsize=(12, 8))
write_data_for_box = [write_times['Baseline (No TensorRT)'], write_times['TensorRT FP32'], write_times['TensorRT FP16']]
bp = plt.boxplot(write_data_for_box, tick_labels=labels, patch_artist=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add text annotations
for i, data in enumerate(write_data_for_box, 1):
    median = np.median(data)
    mean = data.mean()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    plt.text(i, median, f'{median:.3f}', ha='center', va='center', fontsize=10, 
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
             edgecolor='black', alpha=0.8))
    
    plt.text(i - 0.15, mean, f'μ: {mean:.3f}', ha='right', va='center', 
             fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor='lightyellow', edgecolor='gray', alpha=0.7))
    
    plt.text(i + 0.4, q3, f'Q3: {q3:.3f}', ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
             edgecolor='gray', alpha=0.7))
    
    plt.text(i + 0.4, q1, f'Q1: {q1:.3f}', ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
             edgecolor='gray', alpha=0.7))

plt.ylabel('Write Time (s)', fontsize=12)
plt.title('Write Time Distribution Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "write_time_boxplot.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'write_time_boxplot.png'}")

# Stacked bar chart showing time breakdown for averages
plt.figure(figsize=(12, 8))
x = np.arange(len(runs))
width = 0.6

for idx, (run_id, run_name) in enumerate(runs.items()):
    load_avg = data_df[f"{run_id}_bsq_load_time"].mean()
    inference_avg = data_df[f"{run_id}_inference_time"].mean()
    write_avg = data_df[f"{run_id}_write_time"].mean()
    
    # Plot bars
    bar1 = plt.bar(x[idx], load_avg, width, label='Load' if idx == 0 else '', 
            color='#ff7f0e', alpha=0.8)
    bar2 = plt.bar(x[idx], inference_avg, width, bottom=load_avg,
            label='Inference' if idx == 0 else '', color='#2ca02c', alpha=0.8)
    bar3 = plt.bar(x[idx], write_avg, width, bottom=load_avg + inference_avg,
            label='Write' if idx == 0 else '', color='#1f77b4', alpha=0.8)
    
    # Add value labels
    plt.text(x[idx], load_avg / 2, f'{load_avg:.2f}s', ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(x[idx], load_avg + inference_avg / 2, f'{inference_avg:.2f}s', ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(x[idx], load_avg + inference_avg + write_avg / 2, f'{write_avg:.2f}s', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add total at the top
    total = load_avg + inference_avg + write_avg
    plt.text(x[idx], total + 5, f'Total: {total:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.ylabel('Average Time (s)', fontsize=12)
plt.title('Average Time Breakdown by Stage', fontsize=14, fontweight='bold')
plt.xticks(x, [runs[r] for r in runs.keys()], rotation=15, ha='right')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "time_breakdown_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'time_breakdown_comparison.png'}")

# Generate detailed statistics CSV
stats_data = []
for run_id, run_name in runs.items():
    inference_col = f"{run_id}_inference_time"
    load_col = f"{run_id}_bsq_load_time"
    write_col = f"{run_id}_write_time"
    total_col = f"{run_id}_total_time"
    
    stats_data.append({
        'Run': run_name,
        'Avg Load (s)': data_df[load_col].mean(),
        'Median Load (s)': np.median(data_df[load_col]),
        'Std Load (s)': data_df[load_col].std(),
        'Avg Inference (s)': data_df[inference_col].mean(),
        'Median Inference (s)': np.median(data_df[inference_col]),
        'Std Inference (s)': data_df[inference_col].std(),
        'Avg Write (s)': data_df[write_col].mean(),
        'Median Write (s)': np.median(data_df[write_col]),
        'Std Write (s)': data_df[write_col].std(),
        'Avg Total (s)': data_df[total_col].mean(),
        'Median Total (s)': np.median(data_df[total_col]),
        'Std Total (s)': data_df[total_col].std()
    })

stats_df = pd.DataFrame(stats_data)
stats_df.to_csv(output_dir / "detailed_statistics.csv", index=False)
print(f"✓ Saved: {output_dir / 'detailed_statistics.csv'}")

# Speedup statistics CSV
speedup_data = {
    'Sample': data_df['sample_id'].values,
    'Baseline Inference (s)': baseline,
    'FP32 Inference (s)': inference_times['TensorRT FP32'],
    'FP16 Inference (s)': inference_times['TensorRT FP16'],
    'FP32 Speedup': fp32_speedup,
    'FP16 Speedup': fp16_speedup
}
speedup_df = pd.DataFrame(speedup_data)
speedup_df.to_csv(output_dir / "speedup_analysis.csv", index=False)
print(f"✓ Saved: {output_dir / 'speedup_analysis.csv'}")

print("\n" + "=" * 80)
print(f"Analysis complete! All graphs saved to: {output_dir}")
print("=" * 80)
