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
```
.
├── main.py                 # Primary training script for Resnet18
├── main_resnet50.py        # Training script for Resnet50
├── cnn_model.py            # Model definitions
├── data/                   # Dataset handling
├── deepspeed_config.json   # DeepSpeed configuration
├── environment.yml         # Conda environment specification
├── Bandwidth Analysis/     # Performance analysis folder
└── outputFolder/           # Experiment outputs
```

## Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate federated_learning_env
```

## Execution Commands
```bash
# Train Resnet18
python main.py

# Train Resnet50
python main_resnet50.py

# Run with DeepSpeed optimization
deepspeed main.py
```

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
