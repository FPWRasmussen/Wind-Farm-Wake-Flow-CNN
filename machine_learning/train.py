import os
import sys
import yaml
import time
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from CustomLoss import WeightedLoss
from data_handler.dataset import PyWakeDataset, ToTensor, GraphDataset
from models.CAE import CAE
from torch.utils.data import DataLoader
from machine_learning.models.CAE3 import CAE3

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(CUR_DIR)
sys.path.append(PROJ_DIR)

class ConfigLoader:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        self._config = config_dict

    def __getattr__(self, attr):
        value = self._config.get(attr, {})
        if isinstance(value, dict):
            return ConfigLoader(value)
        else:
            return value

    def __repr__(self):
        return repr(self._config)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(config_dict)


def normalize(x, stats, device):
    ws = stats["wind_speed"].view(-1, 1, 1).to(device)
    # ti = stats["turbulence_intensity"].view(-1, 1, 1).to(device)
     
    x[:, 0, :, :] = 1 - torch.clamp(1 - (x[:, 0, :, :] / ws), min=0, max=1)**(1/5)
    # x[:, 1, :, :] = torch.clamp(1 - ti/x[:, 1, :, :], min=0, max=1)**(1/2)
    # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # x[:, 0, :, :] = 1 - x[:, 0, :, :] / ws
    # x[:, 1, :, :] = x[:, 1, :, :] - ti
    return x

def train(config, model, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"PyTorch Model: {model.__class__.__name__}")
    print(f"Using device: {device}")
    print(f"Loss Functions: {type(criterion).__name__} | Optimizer: {type(optimizer).__name__}")
    print("Optimizer parameters:")
    optimizer_params = optimizer.param_groups[0].copy()
    del optimizer_params["params"]
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    print(optimizer_params)
    print(f"Total Epochs: {config.hyperparameters.epochs}")
    
    model.to(device)  # Move the model to the device
    
    # initialize the datasets and dataloaders
    train_dataset = PyWakeDataset(config.io_settings.train_dataset_path, transform=ToTensor(["WS_eff"], ["WS", "TI"])) 
    train_loader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, pin_memory=True, num_workers=config.run_settings.num_workers_train)
    if config.run_settings.validate:
        test_dataset = PyWakeDataset(config.io_settings.test_dataset_path, transform=ToTensor(["WS_eff"], ["WS", "TI"]))   
        test_loader = DataLoader(test_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, pin_memory=True, num_workers=config.run_settings.num_workers_test)
        
    if config.io_settings.pretrained_model: 
        model.load_state_dict(torch.load(config.io_settings.pretrained_model, map_location=torch.device(device))["model_state_dict"])

    if config.hyperparameters.scheduler:
        if config.hyperparameters.scheduler == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.hyperparameters.epochs, eta_min=0)
        elif config.hyperparameters.scheduler == "LinearLR":
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0, total_iters=config.hyperparameters.epochs)
        else:
            raise KeyError("Invalid scheduler name. Please use 'CosineAnnealingLR', 'LinearLR' or leave the string empty.")
    
    training_progress = []
    for epoch in range(config.hyperparameters.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        start_time = time.time()
        train_loss_cum = 0
        model.train()    
        
        for i_batch, train_batch in enumerate(train_loader):
            stats, grid, output, labels = train_batch.values()
            
            # Move data to the device
            grid = grid.to(device)
            output = output.to(device)
            labels = labels.to(device)
            
            y_pred = model(grid, labels)
            output = normalize(output, stats, device)
            train_loss = criterion(y_pred, output)
            train_loss_cum += train_loss.item()

            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()
        
        if config.run_settings.validate:
            test_loss_cum = 0
            model.eval()
            with torch.inference_mode():
                for i_batch, test_batch in enumerate(test_loader):
                    stats, grid, output, labels = test_batch.values()
                    
                    grid = grid.to(device)
                    output = output.to(device)
                    labels = labels.to(device)
                    
                    y_pred = model(grid, labels)
                    output = normalize(output, stats, device)
                    test_loss = criterion(y_pred, output)
                    test_loss_cum += test_loss.item()
            test_loss_cum = test_loss_cum / len(test_loader)
 
        train_loss_cum = train_loss_cum / len(train_loader)
        
        epoch_time = time.time() - start_time
        training_progress.append([epoch, train_loss.item(), train_loss_cum, test_loss.item(), test_loss_cum, epoch_time, current_lr])

        if epoch < config.hyperparameters.lr_decay_stop:
            scheduler.step()
            # for param_group in optimizer.param_groups:
            #     print(f"Epoch {epoch + 1}, Current learning rate: {param_group['lr']}")
            
        if (epoch + 1) % config.io_settings.save_epochs == 0 or (epoch + 1) == config.hyperparameters.epochs:
            model_name = f"model_{model.__class__.__name__}_{optimizer_params['lr']}_e{epoch + 1}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            os.makedirs(config.io_settings.saved_models_dir, exist_ok=True)
            model_saved_path = os.path.join(config.io_settings.saved_models_dir, model_name)
            torch.save({'model_state_dict': model.state_dict(),
                        'training_progress' : training_progress}, 
                       model_saved_path+".pth")

        print(f"Epoch: {epoch+1}/{config.hyperparameters.epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.5e} | Loss: {train_loss:.5e} | Test Loss: {test_loss:.5e}") 

if __name__ == "__main__":
    config = ConfigLoader.from_yaml("./machine_learning/config.yaml") 
    model = CAE3(config.model_settings.input_layers, config.model_settings.output_layers, config.model_settings.latent_space_layers)
    optimizer = optim.Adam(model.parameters(), lr=float(config.hyperparameters.start_lr))
    criterion = WeightedLoss(loss_type="mse", split_ratio=[25, 75],weights=[0.1, 0.9])

    train(config, model, optimizer, criterion)