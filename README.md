# Brain-CEL Spiking Neural Network Framework

## Installation

### System Dependencies

```bash
sudo apt update
sudo apt install cmake python3 python3-venv python3-dev build-essential
```

> **CUDA Toolkit**: Install the latest NVIDIA CUDA Toolkit from the official website:  
> [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### Build Instructions

1. Navigate to the project root
   
   ```
   cd /path/to/BrainCEL
   ```
2. Create build directory and compile
   
   ```bash
   mkdir -p build && cd build
   cmake ..
   cmake --build . -j$(nproc)
   ```

---

## Reproducing the Paper's Results

The `examples/` directory contains scripts and executables to reproduce the evaluation figures from the paper.

### Figure 3 (Brain-CEL vs. NEST)

1. Navigate to example directory in build:
   
   ```shell
   cd examples/nestcompare
   ```

2. Run the Brain-CEL benchmark:
   
   ```shell
   ./nestcompare
   ```
   
   → Output: `braincel_bench.txt`

3. Run the NEST benchmark:
   
   ```shell
   python nest_bench.py
   ```
   
   → Output: `nest_bench.txt`

4. Visualize the data:
   
   ```bash
   python visualize.py
   ```

### Figure 4 (STDP vs. FT-STDP)

1. Navigate to example directory in build:
   
   ```shell
   cd examples/ftcompare
   ```

2. Run the benchmark:
   
   ```shell
   ./ftcompare
   ```
   
   → Output: `braincel_bench.txt`

3. Visualize the data:
   
   ```bash
   python visualize.py
   ```

---

## License

BrainCEL is licensed under the **GNU Affero General Public License v3.0 or later**.
See the full license text  in [LICENSE](LICENSE).