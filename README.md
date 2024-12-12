# Efficient Federated Learning using Gradient-Based Pruning and Adaptive Federated Optimization

## Project Description
This research project addresses critical challenges in Federated Learning (FL), specifically focusing on:
- Reducing high communication costs
- Handling diverse client capabilities
- Improving model convergence in distributed GPU environments

## Project Milestones and Completion Status
- [x] Implement Gradient-Based Pruning
- [x] Develop Adaptive Federated Optimization (AFO)
- [x] Benchmark Performance on CIFAR-10 Dataset
- [x] Analyze Communication Efficiency
- [x] Compare Different Compression Techniques

## Repository Structure
## Repository Structure
```bash
.
├── main.py                     # Experiments 0 & 2: Baseline and Compression
│   ├── deepspeed_config_exp0.json  # Baseline experiment configuration
│   └── deepspeed_config_exp2.json  # Compression experiment configuration
├── main_fp16.py                # FP16 + Compression experiment
│   └── deepspeed_config_exp1.json  # FP16 experiment configuration
├── cnn_model.py                # Model definitions
├── data/                       # Dataset handling
├── environment.yml             # Conda environment specification
├── Bandwidth Analysis/         # Data extraction, analysis scripts, CSVs, and plots
└── outputFolder/               # Raw logging files, network information, train-test errors, and accuracies

## Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate federated_learning_env
```


## Main Results

### Comprehensive Results Table

| Experiment | Model | Compression | Precision | Bandwidth Efficiency | Test Accuracy | Convergence Speed | Communication Overhead | Gradient Update Frequency |
|-----------|-------|-------------|-----------|---------------------|--------------|------------------|------------------------|--------------------------|
| Baseline | ResNet18 | No | FP32 | Baseline | 93.5% | Baseline | High | Standard |
| Exp 0: Compression | ResNet18 | Sparsification + Quantization | FP32 | High | 93.2% | Improved | Reduced | Standard |
| Exp 1: FP16 | ResNet18 | FP16 + Compression | FP16 | Moderate | 92.8% | Accelerated | Moderate | Increased |
| Baseline | ResNet50 | No | FP32 | Baseline | 94.1% | Baseline | High | Standard |
| Exp 0: Compression | ResNet50 | Sparsification + Quantization | FP32 | High | 93.9% | Improved | Reduced | Standard |
| Exp 1: FP16 | ResNet50 | FP16 + Compression | FP16 | Moderate | 93.6% | Accelerated | Moderate | Increased |

## Experimental Results and Observations

### Performance Metrics
| Technique | Bandwidth Efficiency | Model Accuracy | Convergence Speed |
|-----------|---------------------|---------------|------------------|
| Gradient Compression | High | Stable | Improved |
| FP16 + Compression | Moderate | Slightly Reduced | Variable |

### Key Findings
1. **Gradient Compression Advantages**:
   - Lower communication payload
   - No precision loss
   - Stable model convergence

2. **Comparative Analysis**:
   - Full precision gradients perform better than FP16
   - Avoid cumulative approximation errors
   - More effective in bandwidth-constrained setups

### Visualization
Refer to experiment charts in the repository:
- `HPML exp1.png`: Experiment 1 Results
- `HPML exp2.png`: Experiment 2 Results

## Technology Stack
- PyTorch
- Hugging Face Accelerate
- DeepSpeed
- NVIDIA NCCL
- Distributed GPU Training

## Limitations
- Simulated client environments
- Uniform dataset distribution
- Simplified network topology

## Future Work
- Implement more diverse client scenarios
- Explore non-uniform data distribution techniques
- Enhance gradient compression algorithms
