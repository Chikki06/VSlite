# Add these imports at the top
import time
import torch_tensorrt
import tensorrt as trt
from torch.jit import script
import argparse
import os
import json
from datetime import datetime

# Standalone precompilation script
def precompile_tensorrt_engines():
    """Precompile TensorRT engines for all model configurations"""
    
    parser = argparse.ArgumentParser(description='Precompile TensorRT engines')
    parser.add_argument('--model_types', nargs='+', type=int, default=[0, 1], choices=[0, 1],
                       help='Model types to precompile (default: both 0 and 1)')
    parser.add_argument('--precisions', nargs='+', type=str, default=['fp16'], 
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precisions to precompile (default: fp16)')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Batch sizes to precompile (default: 1,2,4,8)')
    parser.add_argument('--workspace_size', type=int, default=1024,
                       help='TensorRT workspace size in MB (default: 1024)')
    parser.add_argument('--force_rebuild', action='store_true', default=False,
                       help='Force rebuild even if engine exists')
    parser.add_argument('--save_timing', action='store_true', default=False,
                       help='Save compilation timing information')
    
    args = parser.parse_args()
    
    # Model configurations
    model_configs = {
        0: {'channels': 10, 'fact': 1472, 'model_file': '20_net_G.pth'},
        1: {'channels': 13, 'fact': 1024, 'model_file': '15_net_G.pth'}
    }
    
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timing_results = []
    
    print("=== TensorRT Engine Precompilation ===")
    print(f"Model types: {args.model_types}")
    print(f"Precisions: {args.precisions}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Workspace size: {args.workspace_size} MB")
    print("=" * 50)
    
    for model_type in args.model_types:
        config = model_configs[model_type]
        model_path = os.path.join(PARENT_DIR, 'saved_models', config['model_file'])
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        print(f"\nProcessing Model Type {model_type} ({config['channels']} channels)")
        print("-" * 40)
        
        # Load base model
        netG = AttGenerator_ResNet50_SkipCon(input_channels=config['channels'])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        netG.load_state_dict(torch.load(model_path))
        netG.to(device)
        netG.eval()
        
        for precision in args.precisions:
            for batch_size in args.batch_sizes:
                input_shape = (batch_size, config['channels'], config['fact'], config['fact'])
                
                # Generate engine path
                engine_path = model_path.replace('.pth', f'_trt_{model_type}_{precision}_bs{batch_size}.ts')
                
                # Check if engine already exists
                if os.path.exists(engine_path) and not args.force_rebuild:
                    print(f"  ✓ Engine exists: {precision} BS{batch_size} -> {os.path.basename(engine_path)}")
                    continue
                
                print(f"  🔄 Compiling: {precision} BS{batch_size}...", end=" ")
                
                try:
                    # Time the compilation
                    compile_start = time.time()
                    
                    # Create dummy input for tracing
                    dummy_input = torch.randn(input_shape).cuda()
                    
                    # Trace the model
                    traced_model = torch.jit.trace(netG, dummy_input)
                    
                    # Set precision
                    if precision == 'fp16':
                        enabled_precisions = {torch.half}
                    elif precision == 'int8':
                        enabled_precisions = {torch.int8}
                    else:
                        enabled_precisions = {torch.float}
                    
                    # Convert to TensorRT
                    trt_model = torch_tensorrt.compile(
                        traced_model,
                        inputs=[torch_tensorrt.Input(
                            min_shape=input_shape,
                            opt_shape=input_shape,
                            max_shape=input_shape,
                            dtype=torch.float32
                        )],
                        enabled_precisions=enabled_precisions,
                        workspace_size=args.workspace_size * 1024 * 1024,
                        max_batch_size=batch_size,
                        use_fast_math=True,
                        device=device
                    )
                    
                    compile_time = time.time() - compile_start
                    
                    # Save the engine
                    torch.jit.save(trt_model, engine_path)
                    
                    print(f"✓ {compile_time:.1f}s")
                    
                    # Store timing information
                    timing_info = {
                        'timestamp': datetime.now().isoformat(),
                        'model_type': model_type,
                        'precision': precision,
                        'batch_size': batch_size,
                        'input_shape': input_shape,
                        'compile_time': compile_time,
                        'engine_path': engine_path,
                        'workspace_size_mb': args.workspace_size
                    }
                    timing_results.append(timing_info)
                    
                except Exception as e:
                    print(f"✗ Failed: {str(e)}")
                    continue
    
    # Save timing results
    if args.save_timing and timing_results:
        timing_file = os.path.join(PARENT_DIR, 'tensorrt_compilation_timing.json')
        with open(timing_file, 'w') as f:
            json.dump(timing_results, f, indent=2)
        print(f"\nTiming results saved to: {timing_file}")
    
    print("\n" + "=" * 50)
    print("Precompilation completed!")
    
    if timing_results:
        total_time = sum(r['compile_time'] for r in timing_results)
        print(f"Total compilation time: {total_time:.1f}s")
        print(f"Average compilation time: {total_time/len(timing_results):.1f}s")
        
        # Show breakdown by precision
        precision_times = {}
        for result in timing_results:
            prec = result['precision']
            if prec not in precision_times:
                precision_times[prec] = []
            precision_times[prec].append(result['compile_time'])
        
        print("\nCompilation time by precision:")
        for prec, times in precision_times.items():
            print(f"  {prec}: avg {sum(times)/len(times):.1f}s (min {min(times):.1f}s, max {max(times):.1f}s)")

# Enhanced conversion function with timing
def convert_to_tensorrt_with_timing(model, input_shape, precision='fp16', workspace_size=1024):
    """Convert PyTorch model to TensorRT with detailed timing"""
    print(f"Converting model to TensorRT with {precision} precision...")
    
    # Time each phase
    phases = {}
    
    # Phase 1: Model tracing
    trace_start = time.time()
    dummy_input = torch.randn(input_shape).cuda()
    traced_model = torch.jit.trace(model, dummy_input)
    phases['tracing'] = time.time() - trace_start
    
    # Phase 2: TensorRT compilation
    compile_start = time.time()
    
    # Set precision
    if precision == 'fp16':
        enabled_precisions = {torch.half}
    elif precision == 'int8':
        enabled_precisions = {torch.int8}
    else:
        enabled_precisions = {torch.float}
    
    # Convert to TensorRT
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(
            min_shape=input_shape,
            opt_shape=input_shape,
            max_shape=input_shape,
            dtype=torch.float32
        )],
        enabled_precisions=enabled_precisions,
        workspace_size=workspace_size * 1024 * 1024,
        max_batch_size=input_shape[0],
        use_fast_math=True,
        device=torch.device("cuda:0")
    )
    
    phases['compilation'] = time.time() - compile_start
    phases['total'] = phases['tracing'] + phases['compilation']
    
    # Print detailed timing
    print(f"  Tracing time: {phases['tracing']:.2f}s")
    print(f"  Compilation time: {phases['compilation']:.2f}s")
    print(f"  Total conversion time: {phases['total']:.2f}s")
    print("TensorRT conversion completed successfully!")
    
    return trt_model, phases

# Enhanced engine management with timing
def save_tensorrt_engine_with_timing(trt_model, model_path, model_type, precision, batch_size, timing_info=None):
    """Save TensorRT engine with timing metadata"""
    engine_path = model_path.replace('.pth', f'_trt_{model_type}_{precision}_bs{batch_size}.ts')
    
    save_start = time.time()
    torch.jit.save(trt_model, engine_path)
    save_time = time.time() - save_start
    
    print(f"TensorRT engine saved to: {engine_path} (save time: {save_time:.2f}s)")
    
    # Save metadata
    metadata = {
        'creation_time': datetime.now().isoformat(),
        'model_type': model_type,
        'precision': precision,
        'batch_size': batch_size,
        'save_time': save_time,
        'engine_path': engine_path
    }
    
    if timing_info:
        metadata.update(timing_info)
    
    metadata_path = engine_path.replace('.ts', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return engine_path

# Smart engine loader with timing
def load_tensorrt_engine_with_timing(engine_path):
    """Load TensorRT engine with timing and metadata"""
    if not os.path.exists(engine_path):
        return None, None
    
    load_start = time.time()
    trt_model = torch.jit.load(engine_path)
    load_time = time.time() - load_start
    
    print(f"Loading TensorRT engine from: {engine_path} (load time: {load_time:.2f}s)")
    
    # Try to load metadata
    metadata_path = engine_path.replace('.ts', '_metadata.json')
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Engine created: {metadata.get('creation_time', 'Unknown')}")
        print(f"  Original compilation time: {metadata.get('compilation', 'Unknown')}s")
    
    return trt_model, metadata

# Benchmarking function
def benchmark_tensorrt_engines():
    """Benchmark all available TensorRT engines"""
    parser = argparse.ArgumentParser(description='Benchmark TensorRT engines')
    parser.add_argument('--warmup_runs', type=int, default=10,
                       help='Number of warmup runs (default: 10)')
    parser.add_argument('--benchmark_runs', type=int, default=100,
                       help='Number of benchmark runs (default: 100)')
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save benchmark results to JSON')
    
    args = parser.parse_args()
    
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved_models_dir = os.path.join(PARENT_DIR, 'saved_models')
    
    # Find all TensorRT engines
    engine_files = [f for f in os.listdir(saved_models_dir) if f.endswith('_trt_*.ts')]
    
    if not engine_files:
        print("No TensorRT engines found. Run precompilation first.")
        return
    
    print(f"Found {len(engine_files)} TensorRT engines")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Benchmark runs: {args.benchmark_runs}")
    print("=" * 50)
    
    benchmark_results = []
    
    for engine_file in engine_files:
        engine_path = os.path.join(saved_models_dir, engine_file)
        print(f"\nBenchmarking: {engine_file}")
        
        try:
            # Load engine
            trt_model, metadata = load_tensorrt_engine_with_timing(engine_path)
            
            if metadata:
                batch_size = metadata['batch_size']
                model_type = metadata['model_type']
                precision = metadata['precision']
                
                # Determine input shape
                channels = 13 if model_type == 1 else 10
                fact = 1024 if model_type == 1 else 1472
                input_shape = (batch_size, channels, fact, fact)
            else:
                print("  Warning: No metadata found, skipping...")
                continue
            
            # Create dummy input
            dummy_input = torch.randn(input_shape).cuda()
            
            # Warmup
            print(f"  Warming up ({args.warmup_runs} runs)...")
            with torch.no_grad():
                for _ in range(args.warmup_runs):
                    _ = trt_model(dummy_input)
            
            # Benchmark
            print(f"  Benchmarking ({args.benchmark_runs} runs)...")
            times = []
            with torch.no_grad():
                for _ in range(args.benchmark_runs):
                    torch.cuda.synchronize()
                    start = time.time()
                    _ = trt_model(dummy_input)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"  Results: {avg_time*1000:.2f}ms avg, {min_time*1000:.2f}ms min, {max_time*1000:.2f}ms max")
            
            # Store results
            result = {
                'engine_file': engine_file,
                'model_type': model_type,
                'precision': precision,
                'batch_size': batch_size,
                'input_shape': input_shape,
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'throughput_fps': batch_size / avg_time,
                'benchmark_timestamp': datetime.now().isoformat()
            }
            benchmark_results.append(result)
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    # Save results
    if args.save_results and benchmark_results:
        results_file = os.path.join(PARENT_DIR, 'tensorrt_benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\nBenchmark results saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    if benchmark_results:
        # Group by precision
        precision_results = {}
        for result in benchmark_results:
            prec = result['precision']
            if prec not in precision_results:
                precision_results[prec] = []
            precision_results[prec].append(result)
        
        for precision, results in precision_results.items():
            print(f"\n{precision.upper()} Results:")
            for result in sorted(results, key=lambda x: x['avg_time_ms']):
                print(f"  BS{result['batch_size']} Model{result['model_type']}: "
                      f"{result['avg_time_ms']:.2f}ms avg, {result['throughput_fps']:.1f} FPS")

if __name__ == '__main__':
    # You can run this as a standalone script for precompilation
    precompile_tensorrt_engines()