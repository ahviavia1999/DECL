
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from dataset import PretrainModelDataset
from model_new import Pretrain_Model
from loss import NT_Xent, FeatureDecouplingLoss
from torch_geometric.data import Batch
from rdkit import RDLogger
import os


RDLogger.DisableLog('rdApp.*')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DynamicWeightAverage(nn.Module):
    def __init__(self, num_tasks=3, temp=0.5):
        super(DynamicWeightAverage, self).__init__()
        self.num_tasks = num_tasks
        self.temp = temp
        self.weights = None
        self.last_loss = None

    def forward(self, losses):

        if self.last_loss is None:
            self.last_loss = torch.tensor(losses).detach()
            self.weights = torch.ones_like(self.last_loss) / self.num_tasks
            return self.weights

        current_loss = torch.tensor(losses).detach()
        loss_ratio = current_loss / (self.last_loss + 1e-8)
        
        w = torch.exp(loss_ratio / self.temp)
        weights = w / w.sum()
        
        self.last_loss = current_loss
        self.weights = weights
        
        return weights

def train(epochs, batch_size, temperature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pre_dataset = PretrainModelDataset('./data/pretrain/zinc15_250K.csv')
    total_samples = len(pre_dataset)
    split1_size = int(0.95 * total_samples)
    split2_size = total_samples - split1_size
    train_data, val_data = random_split(
        pre_dataset, 
        [split1_size, split2_size], 
        generator=torch.Generator()
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    
    model = Pretrain_Model(d_hidden=128, tasks_num=3, dropout=0.2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    
    start_epoch = 0
    
    dwa = DynamicWeightAverage(num_tasks=3,temp=0.2).to(device)
    
    best_val_loss = float('inf')

    checkpoint_path = f'result/pretrain/checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = float(checkpoint['best_val_loss'])
        print(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs + 1):

        train_loss, contrast_weight, pred_weight, dec_weight = train_epoch(model, train_loader, optimizer, device, epoch, epochs, temperature, dwa)
        
        val_loss = validate(model, val_loader, device, temperature, epoch, epochs)

        loss_record_path = 'result/pretrain/loss_record.npy'
        if os.path.exists(loss_record_path):
            loss_record = np.load(loss_record_path, allow_pickle=True).item()
        else:
            loss_record = {'epoch': [], 'train_loss': [], 'val_loss': [], 
                        'contrast_weight': [], 'pred_weight': [], 'dec_weight': [],}

        loss_record['epoch'].append(epoch)
        loss_record['train_loss'].append(train_loss)
        loss_record['val_loss'].append(val_loss)
        loss_record['contrast_weight'].append(contrast_weight)
        loss_record['pred_weight'].append(pred_weight)
        loss_record['dec_weight'].append(dec_weight)

        np.save(loss_record_path, loss_record)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'result/pretrain/best_model.pth')
            print(f"Best model saved with val loss: {val_loss:.8f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, checkpoint_path)

def validate(model, data_loader, device, temperature, epoch, epochs):
    model.eval()
    val_bar = tqdm(data_loader)

    criterion_regression = nn.MSELoss()
    criterion_decoupling = FeatureDecouplingLoss().to(device)

    total_loss = 0.0
    total_num = 0
    
    batch_contrast_losses = []
    batch_pred_losses = []
    batch_dec_losses = []
    
    contrast_weight = 0.3
    pred_weight = 0.3
    dec_weight = 0.3
    
    with torch.no_grad():
        for tem in val_bar:
            batch = Batch().from_data_list(tem)
            batch = batch.to(device)
            
            global_proj, sub_proj, pred = model(batch)

            labels = batch.all_props

            criterion = NT_Xent(global_proj.shape[0], temperature, 1)

            contrast_loss = criterion(global_proj, sub_proj)
            pred_loss = criterion_regression(pred, labels)
            dec_loss = criterion_decoupling(global_proj, sub_proj)
            
            batch_contrast_losses.append(contrast_loss.item())
            batch_pred_losses.append(pred_loss.item())
            batch_dec_losses.append(dec_loss.item())

            if len(batch_contrast_losses) >= 5: 
                weights = torch.tensor([contrast_weight, pred_weight, dec_weight], device=device)
                contrast_weight, pred_weight, dec_weight = weights.cpu().numpy()
                batch_contrast_losses = []
                batch_pred_losses = []
                batch_dec_losses = []
            
            loss = contrast_weight * contrast_loss + pred_weight * pred_loss + dec_weight * dec_loss

            total_num += len(tem)
            total_loss += loss.item() * len(tem)

            val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.8f}'.format(
                epoch, epochs, total_loss / total_num))

    return total_loss / total_num
        
def train_epoch(model, data_loader, train_optimizer, device, epoch, epochs, temperature, dwa):
    model.train()
    train_bar = tqdm(data_loader)

    criterion_regression = nn.MSELoss()
    criterion_decoupling = FeatureDecouplingLoss().to(device)

    total_loss = 0.0
    total_num = 0
    
    batch_contrast_losses = []
    batch_pred_losses = []
    batch_dec_losses = []
    
    contrast_weight = 0.3
    pred_weight = 0.3
    dec_weight = 0.3
    
    for tem in train_bar:
        batch = Batch().from_data_list(tem)
        batch = batch.to(device)
        
        global_proj, sub_proj, pred = model(batch)

        labels = batch.all_props

        criterion = NT_Xent(global_proj.shape[0], temperature, 1)

        contrast_loss = criterion(global_proj, sub_proj)
        pred_loss = criterion_regression(pred, labels)
        dec_loss = criterion_decoupling(global_proj, sub_proj)
        
        batch_contrast_losses.append(contrast_loss.item())
        batch_pred_losses.append(pred_loss.item())
        batch_dec_losses.append(dec_loss.item())
        
        if len(batch_contrast_losses) >= 5: 
            weights = dwa([np.mean(batch_contrast_losses), np.mean(batch_pred_losses), np.mean(batch_dec_losses)])
            contrast_weight, pred_weight, dec_weight = weights.cpu().numpy()
            batch_contrast_losses = []
            batch_pred_losses = []
            batch_dec_losses = []
        
        loss = contrast_weight * contrast_loss + pred_weight * pred_loss + dec_weight * dec_loss

        total_num += len(tem)
        total_loss += loss.item() * len(tem)

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f} CW: {:.4f} PW: {:.4f} DW: {:.4f}'.format(
            epoch, epochs, total_loss / total_num, contrast_weight, pred_weight, dec_weight))

        train_optimizer.zero_grad()
        loss.backward()
    
    if batch_contrast_losses:
        weights = dwa([np.mean(batch_contrast_losses), np.mean(batch_pred_losses), np.mean(batch_dec_losses)])
        contrast_weight, pred_weight, dec_weight = weights.cpu().numpy()

    return total_loss / total_num, contrast_weight, pred_weight, dec_weight

if __name__ == '__main__':
    epochs = 200
    batch_size = 128
    temperature = 0.1

    set_seed(10)

    if not os.path.exists('result/pretrain/'):
        os.makedirs('result/pretrain/')
    train(epochs, batch_size, temperature)