import torch
from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from data_tool import get_atom_and_bond_features, match_smiles,get_mol_property, get_erg_fp, get_maccs_fp, get_pubchem_fp, get_substructure_graph_tensor, get_substructure_seq_tensor
from tqdm import tqdm
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class ModelDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename)
        smiles_list = df.iloc[:, 0].tolist()
        labels_list = [df.iloc[:, i + 1].tolist() for i in range(df.shape[1] - 1)]

        invalid_indices = [] 
        
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:  
                invalid_indices.append(idx)
        
        if invalid_indices:
            valid_indices = set(range(len(smiles_list))) - set(invalid_indices)
            self.smiles_list = [smiles_list[i] for i in sorted(valid_indices)]
            self.labels_list = [[col[i] for i in sorted(valid_indices)] for col in labels_list]
        else:
            self.smiles_list = smiles_list
            self.labels_list = labels_list

        data_list = []
        for idx,smiles in tqdm(enumerate(self.smiles_list)):
            labels = []
            if len(self.labels_list) > 0:
                for i in range(len(self.labels_list)):
                    labels.append(self.labels_list[i][idx])
            x, edge_index, edge_attr = get_atom_and_bond_features(smiles)
            x = torch.tensor(x, dtype=torch.long).view(-1, 9)
            edge_index = torch.tensor(edge_index, dtype=torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long).view(-1, 3)
            maccs_fp = torch.tensor(get_maccs_fp(smiles), dtype=torch.float).unsqueeze(0)
            pubchem_fp = torch.tensor(get_pubchem_fp(smiles), dtype=torch.float).unsqueeze(0)
            erg_fp = torch.tensor(get_erg_fp(smiles), dtype=torch.float).unsqueeze(0)
            graph_sub_list = get_substructure_graph_tensor(smiles)

            data = Data(x=x,
                                  edge_index=edge_index,
                                  edge_attr=edge_attr,
                                  smiles=smiles,
                                  labels=labels,
                                  maccs_fp=maccs_fp,
                                  pubchem_fp=pubchem_fp,
                                  erg_fp=erg_fp,
                                  graph_sub_list=graph_sub_list,
                                  )
            data_list.append(data)

        self.data_list = data_list
        print(f"Total number of samples: {len(self.smiles_list)}")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class PretrainModelDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename)
        smiles_list = df.iloc[:, 0].tolist()

        invalid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_indices.append(idx)
        
        if invalid_indices:
            valid_indices = set(range(len(smiles_list))) - set(invalid_indices)
            self.smiles_list = [smiles_list[i] for i in sorted(valid_indices)]
        else:
            self.smiles_list = smiles_list

        data_list = []
        for idx,smiles in tqdm(enumerate(self.smiles_list)):
            all_props = get_mol_property(smiles)
            x, edge_index, edge_attr = get_atom_and_bond_features(smiles)
            x = torch.tensor(x, dtype=torch.long).view(-1, 9)
            edge_index = torch.tensor(edge_index, dtype=torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long).view(-1, 3)
            all_props = torch.tensor(all_props, dtype=torch.float).unsqueeze(0)
            graph_sub_list = get_substructure_graph_tensor(smiles)

            data = Data(x=x,
                                  edge_index=edge_index,
                                  edge_attr=edge_attr,
                                  smiles=smiles,
                                  all_props = all_props,
                                  graph_sub_list=graph_sub_list,
                                  )
            data_list.append(data)

        self.data_list = data_list
        print(f"Total number of samples: {len(self.smiles_list)}")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class PretrainModelDataset_1(Dataset):
    def __init__(self, smiles_list):

        invalid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_indices.append(idx)
        
        if invalid_indices:
            valid_indices = set(range(len(smiles_list))) - set(invalid_indices)
            self.smiles_list = [smiles_list[i] for i in sorted(valid_indices)]
        else:
            self.smiles_list = smiles_list

        data_list = []
        for idx,smiles in tqdm(enumerate(self.smiles_list)):
            all_props = get_mol_property(smiles)
            x, edge_index, edge_attr = get_atom_and_bond_features(smiles)
            x = torch.tensor(x, dtype=torch.long).view(-1, 9)
            edge_index = torch.tensor(edge_index, dtype=torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long).view(-1, 3)
            all_props = torch.tensor(all_props, dtype=torch.float).unsqueeze(0)
            graph_sub_list = get_substructure_graph_tensor(smiles)

            data = Data(x=x,
                                  edge_index=edge_index,
                                  edge_attr=edge_attr,
                                  smiles=smiles,
                                  all_props = all_props,
                                  graph_sub_list=graph_sub_list,
                                  )
            data_list.append(data)

        self.data_list = data_list
        print(f"Total number of samples: {len(self.smiles_list)}")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

if __name__ == "__main__":
    pass
