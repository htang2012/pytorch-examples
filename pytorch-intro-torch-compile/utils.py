"""
Utility functions for torch.compile examples and benchmarking.
"""

import torch
import torch.nn as nn
import time
import functools
from typing import Callable, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from triton.testing import do_bench
import gc

class BenchmarkSuite:
    """Comprehensive benchmarking suite for torch.compile comparisons."""
    
    def __init__(self, warmup_runs: int = 100, benchmark_runs: int = 1000):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
    
    def benchmark_function(self, fn: Callable, name: str, *args, **kwargs) -> float:
        """Benchmark a single function with proper warmup."""
        exec_time = do_bench(
            lambda: fn(*args, **kwargs),
            warmup=self.warmup_runs,
            rep=self.benchmark_runs
        )
        self.results[name] = exec_time
        return exec_time
    
    def compare_eager_vs_compiled(
        self, 
        model: nn.Module, 
        input_data: torch.Tensor,
        compile_modes: List[str] = None
    ) -> Dict[str, float]:
        """Compare eager mode against different compilation modes."""
        if compile_modes is None:
            compile_modes = ["default", "reduce-overhead", "max-autotune"]
        
        results = {}
        
        # Benchmark eager mode
        model.eval()
        eager_time = self.benchmark_function(model, "eager", input_data)
        results["eager"] = eager_time
        
        # Benchmark compiled modes
        for mode in compile_modes:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, mode=mode)
            compiled_time = self.benchmark_function(compiled_model, f"compiled_{mode}", input_data)
            results[f"compiled_{mode}"] = compiled_time
            
            # Calculate speedup
            speedup = eager_time / compiled_time
            results[f"speedup_{mode}"] = speedup
        
        return results
    
    def plot_results(self, title: str = "Performance Comparison"):
        """Plot benchmark results."""
        if not self.results:
            print("No results to plot. Run benchmarks first.")
            return
        
        # Separate timings and speedups
        timings = {k: v for k, v in self.results.items() if not k.startswith("speedup_")}
        speedups = {k: v for k, v in self.results.items() if k.startswith("speedup_")}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot execution times
        names = list(timings.keys())
        times = list(timings.values())
        
        bars1 = ax1.bar(names, times, color=['red', 'blue', 'green', 'purple'][:len(names)])
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title(f'{title} - Execution Times')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time:.2f}', ha='center', va='bottom')
        
        # Plot speedups if available
        if speedups:
            speedup_names = [k.replace("speedup_", "") for k in speedups.keys()]
            speedup_values = list(speedups.values())
            
            bars2 = ax2.bar(speedup_names, speedup_values, color=['orange', 'cyan', 'magenta'][:len(speedups)])
            ax2.set_ylabel('Speedup (x)')
            ax2.set_title(f'{title} - Speedup over Eager')
            ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, speedup in zip(bars2, speedup_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

class MemoryProfiler:
    """Memory profiling utilities for torch.compile."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0
    
    @staticmethod
    def profile_memory(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Profile memory usage of a function."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        initial_memory = MemoryProfiler.get_memory_usage()
        result = func(*args, **kwargs)
        peak_memory = MemoryProfiler.get_memory_usage()
        
        return result, peak_memory - initial_memory
    
    @staticmethod
    def compare_memory(eager_fn: Callable, compiled_fn: Callable, *args, **kwargs) -> Dict[str, float]:
        """Compare memory usage between eager and compiled functions."""
        _, eager_memory = MemoryProfiler.profile_memory(eager_fn, *args, **kwargs)
        _, compiled_memory = MemoryProfiler.profile_memory(compiled_fn, *args, **kwargs)
        
        return {
            "eager_memory_gb": eager_memory,
            "compiled_memory_gb": compiled_memory,
            "memory_reduction_percent": (eager_memory - compiled_memory) / eager_memory * 100 if eager_memory > 0 else 0
        }

class ModelFactory:
    """Factory for creating test models with different characteristics."""
    
    @staticmethod
    def create_mlp(input_dim: int = 1024, hidden_dims: List[int] = None, output_dim: int = 10) -> nn.Module:
        """Create a multi-layer perceptron."""
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_conv_model(num_classes: int = 10) -> nn.Module:
        """Create a simple convolutional model."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    @staticmethod
    def create_rnn_model(vocab_size: int = 1000, embed_dim: int = 128, hidden_dim: int = 256, num_classes: int = 10) -> nn.Module:
        """Create a simple RNN model."""
        class SimpleRNN(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.classifier = nn.Linear(hidden_dim, num_classes)
            
            def forward(self, x):
                embedded = self.embedding(x)
                output, (hidden, _) = self.rnn(embedded)
                # Use last output for classification
                return self.classifier(output[:, -1, :])
        
        return SimpleRNN(vocab_size, embed_dim, hidden_dim, num_classes)

def compilation_timer(func: Callable) -> Callable:
    """Decorator to time compilation and execution separately."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Time compilation
        compile_start = time.time()
        compiled_func = torch.compile(func)
        compile_time = time.time() - compile_start
        
        # Time first execution (includes compilation)
        exec_start = time.time()
        result = compiled_func(*args, **kwargs)
        first_exec_time = time.time() - exec_start
        
        # Time subsequent execution
        exec_start = time.time()
        compiled_func(*args, **kwargs)
        subsequent_exec_time = time.time() - exec_start
        
        print(f"Compilation time: {compile_time:.4f}s")
        print(f"First execution time: {first_exec_time:.4f}s")
        print(f"Subsequent execution time: {subsequent_exec_time:.4f}s")
        
        return result
    return wrapper

def analyze_graph_breaks(model: nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
    """Analyze graph breaks in a compiled model."""
    torch._dynamo.reset()
    
    # Compile with graph break tracking
    compiled_model = torch.compile(model, fullgraph=False)
    
    # Execute to trigger compilation
    _ = compiled_model(input_data)
    
    # Get graph break statistics
    break_reasons = torch._dynamo.utils.counters.get("graph_break", {})
    frame_stats = torch._dynamo.utils.counters.get("frames", {})
    
    analysis = {
        "graph_breaks": dict(break_reasons),
        "frame_stats": dict(frame_stats),
        "total_breaks": sum(break_reasons.values()) if break_reasons else 0
    }
    
    return analysis

def create_sample_data(model_type: str = "mlp", batch_size: int = 32, device: str = "cuda") -> torch.Tensor:
    """Create sample data for different model types."""
    if not torch.cuda.is_available():
        device = "cpu"
    
    if model_type == "mlp":
        return torch.randn(batch_size, 1024, device=device)
    elif model_type == "conv":
        return torch.randn(batch_size, 3, 32, 32, device=device)
    elif model_type == "rnn":
        return torch.randint(0, 1000, (batch_size, 50), device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def print_compilation_summary(results: Dict[str, Any]):
    """Print a formatted summary of compilation results."""
    print("=" * 60)
    print("TORCH.COMPILE BENCHMARK SUMMARY")
    print("=" * 60)
    
    if "eager" in results:
        print(f"Eager mode time: {results['eager']:.4f} ms")
    
    for key, value in results.items():
        if key.startswith("compiled_"):
            mode = key.replace("compiled_", "")
            print(f"Compiled ({mode}): {value:.4f} ms")
            
            speedup_key = f"speedup_{mode}"
            if speedup_key in results:
                speedup = results[speedup_key]
                print(f"  Speedup: {speedup:.2f}x")
    
    print("=" * 60)