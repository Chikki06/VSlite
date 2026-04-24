import argparse
import glob
import os

def plot_tb_logs(log_dir, output_dir):
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("matplotlib or tensorboard wait not found, skipping curve generation")
        return
        
    event_files = glob.glob(os.path.join(log_dir, "*tfevents*"))
    if not event_files:
        # Check subdirectories
        event_files = glob.glob(os.path.join(log_dir, "**", "*tfevents*"), recursive=True)
        if not event_files:
            print(f"No tfevents files found in {log_dir}")
            return
            
    print(f"Found {len(event_files)} event files, extracting scalars...")
    
    # Load all scalars
    scalars = {}
    for event_file in event_files:
        try:
            ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
            ea.Reload()
            tags = ea.Tags()['scalars']
            for tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                if tag not in scalars:
                    scalars[tag] = {"steps": steps, "vals": vals}
                else:
                    scalars[tag]["steps"].extend(steps)
                    scalars[tag]["vals"].extend(vals)
        except Exception as e:
            print(f"Failed to read {event_file}: {e}")
                
    if not scalars:
        print("No scalars found in the event files.")
        return
        
    # Group tags into loss, metrics, other
    losses = {k: v for k, v in scalars.items() if 'loss' in k.lower()}
    metrics = {k: v for k, v in scalars.items() if 'metric' in k.lower() or 'psnr' in k.lower() or 'ssim' in k.lower() or 'fid' in k.lower()}
    
    os.makedirs(output_dir, exist_ok=True)
    
    def plot_group(group_dict, name):
        if not group_dict: return
        plt.figure(figsize=(10, 6))
        for tag, data in group_dict.items():
            steps = data["steps"]
            vals = data["vals"]
            # sort by step
            sorted_pairs = sorted(zip(steps, vals))
            if not sorted_pairs: continue
            steps, vals = zip(*sorted_pairs)
            plt.plot(steps, vals, label=tag, alpha=0.8)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.title(f"{name} Over Time")
        plt.legend(loc='lower right')
        plt.grid(True)
        out_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")
        
    plot_group(losses, "Training Losses")
    plot_group(metrics, "Metrics")
    
    # Plot anything else that isn't in losses or metrics
    other = {k: v for k, v in scalars.items() if k not in losses and k not in metrics}
    if other:
        plot_group(other, "Other Scalars")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing tensorboard logs")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the plots")
    args = parser.parse_args()
    plot_tb_logs(args.log_dir, args.out_dir)