"""
Memory-optimized training script for VSlite
This script includes various memory optimization techniques
"""

import sys
import os
from collections import OrderedDict

# Set memory management before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_MODE"] = "dryrun"

import torch
import numpy as np

# Enable memory optimization features
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import data
import evaluation.group_evaluator as group_evaluator
import util.util as util
from options.train_options import TrainOptions
from trainers import get_trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer

# Memory optimization functions
def clear_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            print(f"GPU {i}: Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")

os.makedirs('images/', exist_ok=True)

# Parse options
opt = TrainOptions().parse()

# Memory optimization: Force smaller settings for tight memory constraints
print(f"Original settings - batch_size: {opt.batch_size}, crop_size: {opt.crop_size}, num_patches: {opt.num_patches}")

# Override settings for memory efficiency
if opt.batch_size > 2:
    opt.batch_size = 1
    print(f"Reduced batch_size to {opt.batch_size} for memory efficiency")

if opt.crop_size > 384:
    opt.crop_size = 256
    opt.load_size = 512
    print(f"Reduced crop_size to {opt.crop_size}, load_size to {opt.load_size}")

if opt.num_patches > 256:
    opt.num_patches = 128
    print(f"Reduced num_patches to {opt.num_patches}")

if opt.ngf > 48:
    opt.ngf = 32
    print(f"Reduced ngf to {opt.ngf}")

if opt.ndf > 48:
    opt.ndf = 32
    print(f"Reduced ndf to {opt.ndf}")

if opt.num_D > 1:
    opt.num_D = 1
    print(f"Reduced num_D to {opt.num_D}")

print(f"Final optimized settings - batch_size: {opt.batch_size}, crop_size: {opt.crop_size}, num_patches: {opt.num_patches}")
    
# Print options to help debugging
print(' '.join(sys.argv))
print_memory_usage()

# Load the dataset
dataloader = data.create_dataloader(opt)
test_dataloader = data.create_dataloader(util.copyconf(opt, phase="test", batch_size=1))

# Create trainer for our model
trainer = get_trainer(opt)

# Create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader.dataset))

# Create tool for visualization
visualizer = Visualizer(opt)

# Create evaluator for evaluation
evaluator = group_evaluator.GroupEvaluator(opt)

print("Initial memory usage after model loading:")
print_memory_usage()

try:
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # Clear cache periodically to prevent memory fragmentation
            if i % 50 == 0:
                clear_cache()

            # Train generator
            if i % opt.D_steps_per_G == 0:
                for _ in range(opt.num_G_steps):
                    try:
                        trainer.run_generator_one_step(data_i, iter=i)
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM during generator step at iteration {i}, clearing cache and retrying...")
                        clear_cache()
                        print_memory_usage()
                        # Try with smaller effective batch size by processing one sample at a time
                        for batch_idx in range(data_i['input'].size(0)):
                            single_data = {}
                            for key, value in data_i.items():
                                if isinstance(value, torch.Tensor):
                                    single_data[key] = value[batch_idx:batch_idx+1]
                                else:
                                    single_data[key] = value
                            trainer.run_generator_one_step(single_data, iter=i)

            # Train discriminator
            try:
                trainer.run_discriminator_one_step(data_i)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during discriminator step at iteration {i}, clearing cache and retrying...")
                clear_cache()
                print_memory_usage()
                # Try with smaller effective batch size
                for batch_idx in range(data_i['input'].size(0)):
                    single_data = {}
                    for key, value in data_i.items():
                        if isinstance(value, torch.Tensor):
                            single_data[key] = value[batch_idx:batch_idx+1]
                        else:
                            single_data[key] = value
                    trainer.run_discriminator_one_step(single_data)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(
                    epoch,
                    iter_counter.epoch_iter,
                    losses,
                    iter_counter.time_per_iter, trainer.old_lr
                )
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                # Save images but with memory management
                with torch.no_grad():
                    np.save('images/target_'+str(iter_counter.total_steps_so_far)+'.npy',
                           data_i['target'].data.cpu().numpy())
                    np.save('images/input_'+str(iter_counter.total_steps_so_far)+'.npy',
                           data_i['input'].data.cpu().numpy())
                    np.save('images/gen_'+str(iter_counter.total_steps_so_far)+'.npy',
                           trainer.get_latest_generated().data.cpu().numpy())

            if iter_counter.needs_saving():
                print(f'saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})')
                trainer.save('latest')
                iter_counter.record_current_iter()
                
                # Print memory usage after saving
                print("Memory usage after saving:")
                print_memory_usage()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print(f'saving the model at the end of epoch {epoch}, iters {iter_counter.total_steps_so_far}')
            trainer.save(f'{epoch}')
            trainer.save(epoch)

except torch.cuda.OutOfMemoryError as e:
    print(f"Out of memory error: {e}")
    print("Current memory usage:")
    print_memory_usage()
    print("\nMemory optimization suggestions:")
    print("1. Reduce batch_size (current: {})".format(opt.batch_size))
    print("2. Reduce crop_size (current: {})".format(opt.crop_size))
    print("3. Reduce num_patches (current: {})".format(opt.num_patches))
    print("4. Use gradient checkpointing")
    print("5. Use mixed precision training")
    raise e

print('Training was successfully finished.')