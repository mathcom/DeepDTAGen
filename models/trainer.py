import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
from models.utils import get_cindex, get_rm2, get_aupr, mse


class DeepDTAGenTrainer:
    def __init__(self, model, device, optimizer, log_dir='./logs'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.mse_loss_fn = nn.MSELoss()
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.history = {'train_loss': [], 'val_loss': [], 'val_ci': [], 'val_mse': []}


    def train(self, train_loader, valid_loader, n_epochs=500, patience=20, 
              ckpt_dir='./ckpt', ckpt_filename='deepdtagen_best.pth'):
        """
        Main training loop with Early Stopping
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        best_val_mse = float('inf')
        early_stop_count = 0

        for epoch in range(1, n_epochs + 1):
            # --- Training Phase ---
            avg_train_loss, train_metrics = self._train_epoch(train_loader, epoch)

            # --- Evaluation Phase ---
            val_metrics = self.evaluate(valid_loader)
            val_loss = val_metrics['total_loss']
            val_mse = val_metrics['mse']
            val_ci = val_metrics['ci']

            # --- Logging ---
            self._log_metrics(epoch, avg_train_loss, train_metrics, val_metrics)
            
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val MSE: {val_mse:.4f} | Val CI: {val_ci:.4f}")

            # --- Early Stopping & Checkpoint ---
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                early_stop_count = 0
                save_path = os.path.join(ckpt_dir, ckpt_filename)
                torch.save(self.model.state_dict(), save_path)
                print(f" -> Best model saved at epoch {epoch} (MSE: {val_mse:.4f})")
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        self.writer.close()
        return self.history


    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_mse = 0
        
        # Tqdm bar creation
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Model Forward (DeepDTAGen returns 4 values)
            prediction, new_drug, lm_loss, kl_loss = self.model(batch)
            
            # Loss Calculation
            target = batch.y.view(-1, 1).float()
            mse_loss = self.mse_loss_fn(prediction, target)
            
            # Combined Loss (KL weight applied)
            loss = kl_loss * 0.001 + mse_loss + lm_loss
            
            # FetterGrad Backward Step
            losses = [loss, mse_loss]
            self.optimizer.ft_backward(losses)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            
            # Progress bar update
            pbar.set_postfix(MSE=mse_loss.item(), KL=kl_loss.item(), LM=lm_loss.item())

        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        
        return avg_loss, {'mse': avg_mse}


    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Thresholds for AUC calculation (based on original code)
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="[Valid]"):
                batch = batch.to(self.device)
                
                prediction, new_drug, lm_loss, kl_loss = self.model(batch)
                
                # Validation Loss (Composite)
                loss = lm_loss + kl_loss # Note: Original code sums these for validation
                total_loss += loss.item() * batch.num_graphs # Batch size adjustment
                
                all_preds.append(prediction.cpu())
                all_labels.append(batch.y.view(-1, 1).cpu())

        # Concatenate all batches
        P = torch.cat(all_preds, 0).numpy().flatten()
        G = torch.cat(all_labels, 0).numpy().flatten()
        
        # Metrics Calculation
        mse_val = mse(G, P)
        ci_val = get_cindex(G, P)
        rm2_val = get_rm2(G, P)
        
        auc_values = []
        for t in thresholds:
            auc = get_aupr(G, P, t)
            auc_values.append(auc)

        return {
            'total_loss': total_loss / len(data_loader.dataset), # Normalized by total samples
            'mse': mse_val,
            'ci': ci_val,
            'rm2': rm2_val,
            'aucs': auc_values
        }


    def _log_metrics(self, epoch, train_loss, train_metrics, val_metrics):
        # Scalar Logging
        self.writer.add_scalar('Loss/Train_Total', train_loss, epoch)
        self.writer.add_scalar('Loss/Train_MSE', train_metrics['mse'], epoch)
        
        self.writer.add_scalar('Loss/Val_Total', val_metrics['total_loss'], epoch)
        self.writer.add_scalar('Metric/Val_MSE', val_metrics['mse'], epoch)
        self.writer.add_scalar('Metric/Val_CI', val_metrics['ci'], epoch)
        self.writer.add_scalar('Metric/Val_RM2', val_metrics['rm2'], epoch)
        
        # History Update
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_metrics['total_loss'])
        self.history['val_ci'].append(val_metrics['ci'])
        self.history['val_mse'].append(val_metrics['mse'])