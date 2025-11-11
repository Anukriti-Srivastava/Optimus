# Optimus: AI-Powered LLVM Pass Optimizer

Optimus is a project that leverages Reinforcement Learning (RL) to optimize LLVM compiler passes. The goal is to find the best sequence of optimization passes to minimize the size of the compiled code or improve its performance.

## Project Structure

- `main.cpp`: The C++ core of the project, which interacts with LLVM.
- `ai_optimizer_bridge.py`: A Python script that acts as a bridge between the C++ LLVM environment and the Python-based RL agent.
- `optimizer_env.py`: Defines the OpenAI Gym environment for the LLVM pass optimization task.
- `train_rl_agent.py`: The script used to train the PPO (Proximal Policy Optimization) agent.
- `evaluate_agent.py`: A script to evaluate the performance of the trained RL agent.
- `run_optimizer.py`: A script to run the optimizer with the trained agent.
- `models/`: This directory contains the trained RL models.
- `llvm_wrapper/`: This directory contains a wrapper for LLVM functionalities.
- `*.c`, `*.ll`: Sample C and LLVM IR files for testing and optimization.

## How it Works

The project uses a C++ program to expose LLVM's optimization passes to a Python environment. An RL agent, built with libraries like Stable Baselines3, learns to select the best sequence of passes to apply to a given LLVM Intermediate Representation (IR) file. The agent is trained to minimize a reward signal, which can be based on code size, execution time, or other metrics.

## Getting Started

### Prerequisites

- C++ compiler (supporting C++17)
- CMake
- Python 3
- LLVM

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/garvit0224bhardwaj/Optimus.git
   cd Optimus
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the C++ components:**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

### Usage

1. **Train the RL agent:**
   ```bash
   python train_rl_agent.py
   ```

2. **Run the optimizer:**
   ```bash
   python run_optimizer.py --input_file <path_to_llvm_ir_file>
   ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
