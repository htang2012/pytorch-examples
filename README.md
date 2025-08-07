# PyTorch Examples: torch.compile Acceleration and Optimization

A comprehensive repository showcasing PyTorch 2.0's `torch.compile()` functionality with practical examples, benchmarks, and advanced optimization techniques across multiple hardware platforms.

## 🚀 Overview

This repository demonstrates how `torch.compile()` accelerates deep learning models through operator fusion, graph optimization, and platform-specific code generation. It includes educational notebooks, performance benchmarks, and production-ready examples for NVIDIA GPUs, AMD ROCm, Intel processors, Habana Gaudi, and TPUs.

## 📚 Repository Structure

### 🎓 pytorch-intro-torch-compile/
**Comprehensive introduction to torch.compile fundamentals**

- `1-toy-benchmarks.ipynb` - Basic performance comparisons and toy examples
- `2-torch-compile-intro.ipynb` - Introduction to compilation modes and usage
- `3-inspecting-compiler-stack.ipynb` - Deep dive into TorchDynamo, AOTAutograd, PrimTorch, and TorchInductor
- `4-nn-example.ipynb` - Neural network compilation examples
- `5-memory-optimization.ipynb` - **NEW** Memory profiling, gradient checkpointing, and memory-efficient compilation
- `6-debugging-compile-errors.ipynb` - **NEW** Comprehensive debugging guide for compilation errors and fallbacks
- `utils.py` - **NEW** Utility functions for benchmarking, memory profiling, and model factories

### 📝 pytorch-compile-blogpost/
**Interactive exploration of torch.compile internals**

- `torch-compile-under-the-hood.ipynb` - Original blog post content with detailed explanations
- `interactive-compiler-exploration.ipynb` - **NEW** Interactive dashboard for exploring compilation behavior
- `performance-deep-dive.ipynb` - **NEW** Advanced performance analysis with kernel fusion, memory patterns, and batch scaling

### ⚡ pytorch-graph-optimization/
**Advanced graph optimization techniques**

- `benchmark_torch-compile_resnet.ipynb` - ResNet model optimization benchmarks
- `graph_optimization_torch_compile.ipynb` - Graph transformation and optimization strategies  
- `inspecting_torch_compile.ipynb` - Graph inspection and analysis tools
- `advanced-optimization-techniques.ipynb` - **NEW** Custom operators, pattern matching, and advanced compilation strategies

### 🐳 docker/
**Multi-platform containerized environments**

- `Dockerfile.gpu` - NVIDIA GPU environment with CUDA support
- `Dockerfile.habana` - Habana Gaudi AI processor environment
- `Dockerfile.xlagpu` - Google TPU/XLA environment
- `Dockerfile.intel` - **NEW** Intel Extension for PyTorch with CPU optimizations
- `Dockerfile.amd` - **NEW** AMD ROCm environment for GPU acceleration
- `start_*_pytorch2` - **NEW** Convenient startup scripts for each environment
- `README.md` - Docker environment documentation

### 📊 Root Level Examples
- `mnist_metric.py` - Complete MNIST training pipeline with metrics and visualization
- `PT2_Backend_Integration.ipynb` - Backend integration examples with comprehensive benchmarks

## 🔧 Key Features

### Performance Optimization
- **Kernel Fusion Analysis** - Understand how operations get fused for better performance
- **Memory Pattern Optimization** - Analyze and optimize memory access patterns
- **Batch Size Scaling** - Study performance characteristics across different batch sizes
- **Multi-Stage Compilation** - Compare compilation strategies and overhead analysis

### Debugging and Analysis Tools
- **Graph Visualization** - Interactive tools to visualize computation graphs
- **Compilation Error Debugging** - Comprehensive error analysis and workaround suggestions
- **Memory Profiling** - Detailed memory usage tracking and optimization
- **Performance Benchmarking** - Advanced benchmarking suites with statistical analysis

### Hardware Platform Support
- **NVIDIA GPUs** - CUDA-optimized containers with latest drivers
- **AMD GPUs** - ROCm support for AMD graphics cards
- **Intel CPUs** - Intel Extension for PyTorch with MKL optimizations
- **Habana Gaudi** - Specialized AI processor support
- **Google TPUs** - XLA compilation and TPU acceleration

## 🚀 Quick Start

### Option 1: Local Installation
```bash
# Clone the repository
git clone <repository-url>
cd pytorch-examples

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jupyter matplotlib seaborn pandas numpy scikit-learn triton transformers datasets
```

### Option 2: Docker Environments
```bash
# NVIDIA GPU environment
./docker/start_gpu_pytorch2

# AMD ROCm environment  
./docker/start_amd_pytorch2

# Intel CPU optimized environment
./docker/start_intel_pytorch2

# Access Jupyter Lab at http://localhost:8888
```

## 📖 Learning Path

### Beginner Track
1. Start with `pytorch-intro-torch-compile/1-toy-benchmarks.ipynb`
2. Learn compilation modes in `2-torch-compile-intro.ipynb`
3. Try the MNIST example: `python mnist_metric.py`

### Intermediate Track
1. Explore compiler internals in `3-inspecting-compiler-stack.ipynb`
2. Learn memory optimization in `5-memory-optimization.ipynb` 
3. Practice debugging in `6-debugging-compile-errors.ipynb`

### Advanced Track
1. Interactive exploration with `interactive-compiler-exploration.ipynb`
2. Advanced optimization in `advanced-optimization-techniques.ipynb`
3. Performance deep dive with `performance-deep-dive.ipynb`

## 🏆 Performance Highlights

### Typical Speedups with torch.compile
- **ResNet50**: 1.5-2x speedup over eager mode
- **Transformer models**: 1.3-1.8x speedup with optimized attention
- **Custom MLPs**: 2-3x speedup with kernel fusion
- **CNN architectures**: 1.4-2.2x speedup with conv-bn-relu fusion

### Memory Optimization Results
- **Gradient checkpointing**: 30-50% memory reduction
- **Kernel fusion**: 15-25% memory savings
- **Optimized compilation modes**: 10-20% memory efficiency gains

## 🔍 How torch.compile Works

The compilation process involves four key technologies:

1. **TorchDynamo** - Graph acquisition through Python bytecode interpretation
2. **AOTAutograd** - Ahead-of-time automatic differentiation for backward pass optimization
3. **PrimTorch** - Operation decomposition into stable primitive operators
4. **TorchInductor** - High-performance code generation for target hardware

### Compilation Modes
- **default**: Fast compilation, moderate optimization
- **reduce-overhead**: Balanced compilation time and performance
- **max-autotune**: Aggressive optimization, longer compilation time

## 🛠️ Development and Contribution

### Running Tests
```bash
# Test basic functionality
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Run benchmark suite
cd pytorch-intro-torch-compile
python -c "from utils import BenchmarkSuite; suite = BenchmarkSuite(); print('Benchmark tools loaded successfully')"
```

### Adding New Examples
1. Follow the existing notebook structure
2. Include comprehensive benchmarks
3. Add utility functions to appropriate modules
4. Update documentation

## 📚 References and Resources

### Official Documentation
- [PyTorch 2.0 Introduction](https://pytorch.org/blog/pytorch-2.0-release/)
- [torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [TorchDynamo Documentation](https://pytorch.org/docs/main/dynamo/)

### Research Papers and Blogs
- [TorchInductor Paper](https://arxiv.org/abs/2401.05317)
- [Accelerating Deep Learning with PyTorch 2.0](https://medium.com/towards-data-science/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26)
- [AMD ROCm torch.compile Blog](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/)

### Hardware-Specific Resources
- [NVIDIA CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Habana Gaudi Documentation](https://docs.habana.ai/)

## 🤝 Community and Support

- **Issues**: Report bugs and request features through GitHub issues
- **Discussions**: Join community discussions for questions and sharing
- **Contributing**: See CONTRIBUTING.md for development guidelines

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🏷️ Tags

`pytorch` `torch-compile` `machine-learning` `deep-learning` `gpu-acceleration` `performance-optimization` `cuda` `rocm` `intel` `habana` `tpu` `docker` `jupyter` `benchmarking`