import os
import torch
import torch.optim as optim
import pickle
from torch_geometric.loader import DataLoader # PyG DataLoader 사용

# 정의한 모듈들 import
from models.network import DeepDTAGen
from models.trainer import DeepDTAGenTrainer
from models.FetterGrad import FetterGrad
from models.utils import Tokenizer, TestbedDataset

def main():
    # --- Hyperparameters ---
    params = {
        'dataset': 'ChEMBL35_Kd',
        'batch_size': 32,
        'lr': 0.0002,
        'n_epochs': 500,
        'patience': 20
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    # --- 5-Fold Cross Validation ---
    for foldname in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
        print(f"\n{'='*20} Start: {foldname} {'='*20}")

        # 1. Directory Setup
        root_dir = os.path.join('data', params['dataset'], foldname)
        dirpath_logs = os.path.join('logs', f"{params['dataset']}+{foldname}")
        dirpath_ckpt = os.path.join('ckpt', f"{params['dataset']}+{foldname}")

        # 2. Tokenizer Load
        vocab_path = os.path.join(root_dir, 'vocab.txt')
        if not os.path.exists(vocab_path):
            print(f"[Error] Vocabulary not found at {vocab_path}")
            continue
            
        with open(vocab_path, 'r') as f:
            vocab = [x.rstrip() for x in f.readlines()]
        tokenizer = Tokenizer(vocab)

        # 3. Data Loader
        try:
            train_data = TestbedDataset(root=root_dir, dataset='train')
            valid_data = TestbedDataset(root=root_dir, dataset='valid')
        except Exception as e:
            print(f"[Error] Data loading failed for {foldname}. Check .pt files.")
            print(e)
            continue

        train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=params['batch_size'], shuffle=False)

        # 4. Model & Optimizer Setup
        model = DeepDTAGen(tokenizer)
        
        # DeepDTAGen uses FetterGrad wrapper
        base_optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        optimizer = FetterGrad(base_optimizer)

        # 5. Training
        trainer = DeepDTAGenTrainer(
            model=model,
            device=device,
            optimizer=optimizer,
            log_dir=dirpath_logs
        )

        history = trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epochs=params['n_epochs'],
            patience=params['patience'],
            ckpt_dir=dirpath_ckpt,
            ckpt_filename=f"deepdtagen_{foldname}_best.pth"
        )
        
        print(f"{'='*20} Finished: {foldname} {'='*20}")

if __name__ == "__main__":
    main()