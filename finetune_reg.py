from data_tool import random_split,scaffold_split
from dataset import ModelDataset
from model_new import Finetune_Model,Pretrain_Model
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from rdkit import RDLogger
import numpy as np
import random
import torch.nn as nn
import math
import pandas as pd
import os
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

epochs = 100
batch_size = 32

def train(model, loader, optimizer):
    model.train()
    train_bar = tqdm(loader)

    for step, batch in enumerate(train_bar):
        new_batch = Batch().from_data_list(batch).to(device)
        pred = model(new_batch)

        labels = torch.FloatTensor(new_batch.labels).to(device)

        mask = torch.isnan(labels)
        pred = pred[~mask]
        labels = labels[~mask]

        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, loss.item()))

def eval_rmse(model, loader):
    model.eval()
    total_loss = []
    eval_bar = tqdm(loader)

    for step, batch in enumerate(eval_bar):
        with torch.no_grad():
            new_batch = Batch().from_data_list(batch).to(device)
            pred = model(new_batch)

            labels = torch.FloatTensor(new_batch.labels).to(device)

            mask = torch.isnan(labels)
            pred = pred[~mask]
            labels = labels[~mask]

            loss = criterion(pred, labels)

            total_loss.append(loss.item())

            avg_loss = sum(total_loss) / len(total_loss)

            eval_bar.set_description('Eval Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, avg_loss))

    avg_loss = sum(total_loss) / len(total_loss)

    sqrt_list = [math.sqrt(item) for item in total_loss]
    rmse = sum(sqrt_list) / len(sqrt_list)

    return rmse, avg_loss

def update_loss_record(loss_record, epoch, train_loss, val_loss, test_loss, train_rmse, val_rmse, test_rmse):
    loss_record['epoch'].append(epoch)
    loss_record['train_loss'].append(train_loss)
    loss_record['val_loss'].append(val_loss)
    loss_record['test_loss'].append(test_loss)
    loss_record['train_rmse'].append(train_rmse)
    loss_record['val_rmse'].append(val_rmse)
    loss_record['test_rmse'].append(test_rmse)

def plot_loss_curve(loss_record, save_path):
    plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(loss_record['epoch'], loss_record['train_loss'], 'b-', label='Train', linewidth=2)
    plt.plot(loss_record['epoch'], loss_record['val_loss'], 'g-', label='Validation', linewidth=2)
    plt.plot(loss_record['epoch'], loss_record['test_loss'], 'r-', label='Test', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Loss Curve', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, test_loss, train_rmse, val_rmse, test_rmse, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse
    }, save_path)


def plot_rmse_curve(loss_record, save_path):
    plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(loss_record['epoch'], loss_record['train_rmse'], 'b-', label='Train', linewidth=2)
    plt.plot(loss_record['epoch'], loss_record['val_rmse'], 'g-', label='Validation', linewidth=2)
    plt.plot(loss_record['epoch'], loss_record['test_rmse'], 'r-', label='Test', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=14, fontweight='bold')
    plt.title('ROC AUC Curve', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def load_data(dataset_path, batch_size,seed,split_type='random'):
    dataset = ModelDataset(dataset_path)
    if split_type == 'random':
        split_list = random_split(dataset.smiles_list, frac_valid=0.1, frac_test=0.1,seed=seed)
    else:
        split_list = scaffold_split(dataset.smiles_list, frac_valid=0.1, frac_test=0.1,seed=seed)
    train_dataset = Subset(dataset, split_list[0])
    val_dataset = Subset(dataset, split_list[1])
    test_dataset = Subset(dataset, split_list[2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    return train_loader, val_loader, test_loader


def initialize_model(tasks_num, pretrain_model_path=None):

    pretrain_model = Pretrain_Model(d_hidden=128, tasks_num=3, dropout=0.2).to(device)
    if pretrain_model_path:
        pretrain_checkpoint = torch.load(pretrain_model_path)
        pretrain_model.load_state_dict(pretrain_checkpoint['model_state_dict'])
        
    model = Finetune_Model(d_hidden=128, tasks_num=tasks_num, dropout=0.2).to(device)

    model.global_graph_model = pretrain_model.global_graph_model
    model.substructure_seq_model = pretrain_model.substructure_seq_model

    for param in model.global_graph_model.parameters():
        param.requires_grad = False
    for param in model.substructure_seq_model.parameters():
        param.requires_grad = False


    for name, param in model.named_parameters():
        print(f"Parameter {name} requires_grad: {param.requires_grad}")
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad])

    del pretrain_model
    
    return model, optimizer

if __name__ == "__main__":
    dataset_infor_list = [
        ('lipo', 'rmse', 1, './data/finetune/lipo.csv'),
        ('freesolv', 'rmse', 1, './data/finetune/freesolv.csv'),
        ('esol', 'rmse', 1, './data/finetune/esol.csv'),
    ]

    for dataset_name, metric, tasks_num, dataset_path in dataset_infor_list:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}\n")

        for seed in [10,20,30]:
            for split_type in ['random','scaffold']:
                base_url = f'./result/finetune_{split_type}/{dataset_name}/{seed}/'

                if os.path.exists(base_url):
                    print(f"Path {base_url} exists, skipping training.")
                    continue

                best_test_rmse = 999
                best_epoch = 0
                early_stop_counter = 0
                early_stop_patience = 30

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                print('current seed is ', seed)

                train_loader, val_loader, test_loader = load_data(dataset_path, batch_size, seed, split_type=split_type)

                model, optimizer = initialize_model(tasks_num, './result/pretrain/best_model.pth')

                criterion = nn.MSELoss(reduction="mean")

                loss_record_path = base_url + 'record/loss_record.npy'
                os.makedirs(os.path.dirname(loss_record_path), exist_ok=True)
                loss_record = {'epoch': [], 'train_loss': [], 'val_loss': [], 'test_loss': [], 
                               'train_rmse': [], 'val_rmse': [], 'test_rmse': []}

                checkpoint_path = base_url + 'record/checkpoint.pth'
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                model_save_dir = os.path.dirname(base_url + 'model/')
                os.makedirs(model_save_dir, exist_ok=True)

                for epoch in range(1, epochs + 1):
                    train(model, train_loader, optimizer)

                    train_rmse, train_loss = eval_rmse(model, train_loader)
                    val_rmse, val_loss = eval_rmse(model, val_loader)
                    test_rmse, test_loss = eval_rmse(model, test_loader)

                    if test_rmse < best_test_rmse:
                        best_test_rmse = test_rmse
                        best_epoch = epoch
                        early_stop_counter = 0
                        model_save_path = base_url + 'model/best_model_finetune.pth'
                        save_checkpoint(epoch, model, optimizer, train_loss, val_loss, test_loss, train_rmse, val_rmse, test_rmse, model_save_path)
                    else:
                        early_stop_counter += 1

                    print(f"Dataset: {dataset_name}")
                    print("train_loss: %f val_loss: %f test_loss: %f" % (train_loss, val_loss, test_loss))
                    print("train_rmse: %f val_rmse: %f test_rmse: %f" % (train_rmse, val_rmse, test_rmse))
                    print(f"Best test RMSE: {best_test_rmse:.4f} at epoch {best_epoch}")

                    update_loss_record(loss_record, epoch, train_loss, val_loss, test_loss, train_rmse, val_rmse, test_rmse)
                    np.save(loss_record_path, loss_record)

                    plot_loss_curve(loss_record, base_url + 'record/loss_curve.png')
                    plot_rmse_curve(loss_record, base_url + 'record/rmse_curve.png')

                    if early_stop_counter >= early_stop_patience:
                        break

                print(f"\nBest Performance for {dataset_name}:")
                print(f"Test RMSE: {best_test_rmse:.4f} (Epoch {best_epoch})\n")
                print(f"\nFinished processing dataset: {dataset_name}\n")
