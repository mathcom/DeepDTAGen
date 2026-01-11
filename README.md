# DeepDTAGen: Refactored & Dockerized Implementation

This repository is a **refactored and modernized implementation** of the DeepDTAGen framework.  
We have migrated the original codebase (which relied on older dependencies) to a robust **Docker-based environment** utilizing **PyTorch 1.13**, **CUDA 11.7**, and **PyTorch Geometric (PyG)**. The code structure has been redesigned using Object-Oriented Programming (OOP) principles for better maintainability and scalability.

## 🚀 Key Improvements

1.  **Modernized Environment**:
    * **Docker & Docker Compose**: Complete environment encapsulation using **Miniforge** (avoiding Anaconda ToS issues).
    * **GPU Acceleration**: Fully supports CUDA 11.7 with specific PyG wheel installation to prevent CPU-fallback errors.
    * **Library Upgrades**: Migrated to PyTorch 1.13.1 and PyTorch Geometric 2.2.0.
2.  **Code Refactoring**:
    * **Trainer Pattern**: Decoupled the training logic into a modular `Trainer` class (handling training loops, validation, early stopping, and checkpointing).
    * **Robust Data Processing**: Implemented `DeepDTAGenDataProcessor` to handle large-scale graph data generation and caching efficiently.
    * **Strict Type Hinting**: Added type hints for better code readability and debugging.
3.  **Monitoring**:
    * **TensorBoard Integration**: Real-time visualization of Loss, MSE, CI (Concordance Index), and RM2 metrics.

## 📂 Directory Structure

```text
.
├── docker-compose.yml       # Service definitions (DeepDTA & TensorBoard)
├── Dockerfile               # Miniforge-based build setup (PyTorch 1.13 + PyG)
├── README.md                # Project documentation
├── train_ChEMBL35_IC50.py   # Main experiment runner (5-Fold CV)
├── models/
│   ├── trainer.py           # Training loop, logging, and checkpoint logic
│   ├── network.py           # DeepDTAGen model architecture
│   ├── datautils.py         # Data preprocessing and graph conversion
│   └── utils.py             # Dataset classes and metric functions
├── data/                    # Dataset directory
├── ckpt/                    # Model checkpoints
└── logs/                    # TensorBoard log storage
```

## 🛠 Installation & Setup

1.  **Prerequisites**
    * Docker & Docker Compose
    * NVIDIA Driver & NVIDIA Container Toolkit (for GPU support)

2.  **Build and Run Containers**
This command builds the image using Miniforge and starts the JupyterLab and TensorBoard services.

```bash
docker compose up -d --build
```

3.  **Access the Container**
To run scripts manually or debug:

```bash
docker exec -it deepdtagen_container /bin/bash
```

## 🏃‍♂️ Usage

1.  **Data Preprocessing**
Use the `DeepDTAGenDataProcessor` to convert raw CSVs into PyTorch Geometric `.pt` format.

```python
from models.datautils import DeepDTAGenDataProcessor

# Example: Process data for a specific fold
processor = DeepDTAGenDataProcessor(data_dir='data/ChEMBL35_IC50/fold1')
processor.process_datasets(['ChEMBL35_IC50'])
```

- **Note**: Processed files are cached in `data/processed/`. If you change the data logic, ensure you delete the old `.pt` files.


2.  **Training**
Run the experiment script. This script defaults to running a 5-Fold Cross-Validation.

```bash
python train_ChEMBL35_IC50.py
```


3. **Monitoring (TensorBoard)**
You can monitor training progress in real-time via your browser:
   * URL: `http://localhost:6006`

**Note on Permissions:** If you see a *"Failed to fetch runs"* error, it is likely a permission issue (files created by root). Run this on your host machine:

```bash
sudo chmod -R 775 logs/
```

## 🐛 Technical Troubleshooting (Post-Mortem)
This section documents critical issues resolved during the development of this environment.

1.  **Protobuf Version Conflict**
    * **Issue**: `TypeError: MessageToJson() ...` error in TensorBoard.
    * **Cause**: Dependency conflict between `fairseq` (requires old protobuf) and `tensorboard` (requires new protobuf). Recent `protobuf 5.x` breaks backward compatibility.
    * **Solution**: Pinned `protobuf==3.20.3` in the Dockerfile.

2.  **PyTorch Geometric CUDA Error**
    * **Issue**: `RuntimeError: Not compiled with CUDA support` even when GPU is available.
    * **Cause**: `pip` fell back to the PyPI version of `torch-scatter`, which is CPU-only.
    * **Solution**: Explicitly enforced installation via the PyG wheel index:

```Dockerfile
RUN pip install torch-scatter -f [https://data.pyg.org/whl/torch-1.13.1+cu117.html](https://data.pyg.org/whl/torch-1.13.1+cu117.html)
```

3.  **Data Unpacking Error**
    * **Issue**: `ValueError: too many values to unpack (expected 2)` during data loading.
    * **Cause**: The processor saved the entire `Dataset` object instead of the `(data, slices)` tuple required by PyG `InMemoryDataset`.
    * **Solution**: Modified `datautils.py` to save `(self.data, self.slices)`.
    
## 📄 Reference
If you use this code or the DeepDTAGen framework, please cite the original paper:
```
Shah, P.M., Zhu, H., Lu, Z. et al. DeepDTAGen: a multitask deep learning framework for drug-target affinity prediction and target-aware drugs generation. Nat Commun 16, 5021 (2025). https://doi.org/10.1038/s41467-025-59917-6
```