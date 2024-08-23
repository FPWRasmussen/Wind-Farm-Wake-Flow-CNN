import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, random_split

# CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJ_DIR = os.path.dirname(CUR_DIR)
# sys.path.append(PROJ_DIR)


class GraphDataset(Dataset):
    def __init__(self, root_dir, output_vars : list, dtype : torch.dtype = torch.float32):
        self.root_dir = root_dir
        self.dtype = dtype
        # self.folder_list = [os.path.join(root, d) for root, dirs, files in os.walk(root_dir) for d in dirs]
        self.folder_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Sort the folder list to ensure consistent ordering
        self.folder_list.sort()
        self.stats = pd.read_csv(os.path.join(self.root_dir, "stats.csv"))
        
        
        self.output_vars = output_vars
    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        flow_map_xr = xr.load_dataset(os.path.join(self.root_dir, folder_name, "flow_map.nc"))
        graph = torch.load(os.path.join(self.root_dir, folder_name,"graph.pt"))

        graph.x = graph.x.type(self.dtype)
        graph.pos = graph.pos.type(self.dtype)
        graph.globals = graph.globals.type(self.dtype)
        
        flow_map_torch = {
            var_name: torch.from_numpy(flow_map_xr[var_name].values.squeeze()).type(self.dtype)
            for var_name in flow_map_xr.variables
        }
        
        flow_map = torch.cat([torch.cat([flow_map_torch[s].unsqueeze(0) for s in self.output_vars], dim=0)], dim=0)

        sample = {"stats": self.stats.iloc[int(folder_name)].to_dict(),
                "graph" : graph,
                "flow_map" : flow_map}

        return sample


import os
import torch
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

class PyWakeDataset(Dataset):
    def __init__(self, root_dir, dtype=torch.float32, transform=None):
        self.root_dir = root_dir
        self.dtype = dtype
        self.transform = transform
        
        # Get all subdirectories in the root directory
        self.folder_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Sort the folder list to ensure consistent ordering
        # self.folder_list.sort()
        self.folder_list.sort(key=lambda x: int(x))
        # Load stats
        self.stats = pd.read_csv(os.path.join(self.root_dir, "stats.csv"))

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        
        flow_map_xr = xr.load_dataset(os.path.join(self.root_dir, folder_name, "flow_map.nc"))
        sim_res_xr = xr.load_dataset(os.path.join(self.root_dir, folder_name, "sim_res.nc"))
        
        flow_map_torch = {
            var_name: torch.from_numpy(flow_map_xr[var_name].values.squeeze()).type(self.dtype)
            for var_name in flow_map_xr.variables
        }
        
        sim_res_torch = {
            var_name: torch.from_numpy(sim_res_xr[var_name].values.squeeze()).type(self.dtype)
            for var_name in sim_res_xr.variables
        }
        
        sample = {
            "stats": self.stats.iloc[int(folder_name)].to_dict(),
            "flow_map": flow_map_torch,
            "sim_res": sim_res_torch
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# class PyWakeDataset(Dataset):
#     def __init__(self, root_dir, dtype = torch.float32, transform = None):
#         self.root_dir = root_dir
#         self.dtype = dtype
#         self.transform = transform
#         self.folder_list = [os.path.join(root, d) for root, dirs, _ in os.walk(root_dir) for d in dirs]
#         self.stats = pd.read_csv(self.root_dir)
#     def __len__(self):
#         return len(self.folder_list)

#     def __getitem__(self, idx):
#         flow_map_xr = xr.load_dataset(os.path.join(self.root_dir, str(idx),"flow_map.nc"))
#         sim_res_xr = xr.load_dataset(os.path.join(self.root_dir, str(idx),"sim_res.nc"))

#         flow_map_torch = {
#             var_name: torch.from_numpy(flow_map_xr[var_name].values.squeeze()).type(self.dtype)
#             for var_name in flow_map_xr.variables
#         }
        
#         sim_res_torch = {
#             var_name: torch.from_numpy(sim_res_xr[var_name].values.squeeze()).type(self.dtype)
#             for var_name in sim_res_xr.variables
#         }

#         sample = {"stats" : self.stats.iloc[idx],
#                 "flow_map" : flow_map_torch,
#                 "sim_res" : sim_res_torch}
        
#         if self.transform:
#             sample = self.transform(sample)

#         return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, output_vars, latent_space_vars):
        self.output_vars = output_vars
        self.latent_space_vars = latent_space_vars
    def __call__(self, sample):
        stats, flow_map, sim_res = sample.values()
       
        grid = position_to_grid(flow_map["x"], flow_map["y"], torch.vstack([sim_res["x"], sim_res["y"]])).unsqueeze(0)
        
        output = torch.cat([torch.cat([flow_map[s].unsqueeze(0) for s in self.output_vars], dim=0)], dim=0)
        latent_space_labels = torch.tensor([sim_res[s] for s in self.latent_space_vars])
        
        sample = {"stats" : stats,
                  "grid" : grid,
                  "output" : output,
                  "labels" : latent_space_labels}
        
        return sample



def position_to_grid(x_range, y_range, positions, device = "cpu"):
    # x_range = torch.tensor(x_range)
    # y_range = torch.tensor(y_range)
    # positions = torch.tensor(positions)
    x_range = x_range#.clone().detach()
    y_range = y_range#.clone().detach()
    positions = positions#.clone().detach()

    num_x_cells = len(x_range)
    num_y_cells = len(y_range)

    # Create an empty 2D grid
    grid = torch.zeros(num_y_cells, num_x_cells, device=device)

    # Iterate over each position
    for x, y in positions.T:
        # Find the nearest grid cell indices for the position
        x_idx = torch.argmin(torch.abs(x_range - x))
        y_idx = torch.argmin(torch.abs(y_range - y))

        # Calculate the distance from the position to the center of the nearest grid cell
        x_dist = x - x_range[x_idx]
        y_dist = y - y_range[y_idx]

        # Calculate the contribution of the position to the grid cell and its neighbors
        x_con = 1 - torch.abs(x_dist) / (x_range[1] - x_range[0])
        y_con = 1 - torch.abs(y_dist) / (y_range[1] - y_range[0])

        if x_dist >= 0 and y_dist >= 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx + 1] += (1 - x_con) * y_con
            grid[y_idx + 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx + 1, x_idx + 1] += (1 - x_con) * (1 - y_con)
        elif x_dist >= 0 and y_dist < 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx + 1] += (1 - x_con) * y_con
            grid[y_idx - 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx - 1, x_idx + 1] += (1 - x_con) * (1 - y_con)
        elif x_dist < 0 and y_dist >= 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx - 1] += (1 - x_con) * y_con
            grid[y_idx + 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx + 1, x_idx - 1] += (1 - x_con) * (1 - y_con)
        elif x_dist < 0 and y_dist < 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx - 1] += (1 - x_con) * y_con
            grid[y_idx - 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx - 1, x_idx - 1] += (1 - x_con) * (1 - y_con)

    return grid

# def position_to_grid(x_range, y_range, positions):

#     num_x_cells = len(x_range)
#     num_y_cells = len(y_range)
    
#     # Create an empty 2D grid
#     grid = np.zeros((num_y_cells, num_x_cells))
    
#     # Iterate over each position
#     for x, y in positions:
#         # Find the nearest grid cell indices for the position
#         x_idx = np.argmin(np.abs(x_range - x))
#         y_idx = np.argmin(np.abs(y_range - y))
        
#         # Calculate the distance from the position to the center of the nearest grid cell
#         x_dist = x - x_range[x_idx]
#         y_dist = y - y_range[y_idx]
        
#         # Calculate the contribution of the position to the grid cell and its neighbors
#         x_con = 1 - abs(x_dist) / (x_range[1] - x_range[0])
#         y_con = 1 - abs(y_dist) / (y_range[1] - y_range[0])
        

#         if x_dist >= 0 and y_dist >= 0:
#             grid[y_idx, x_idx] += x_con * y_con
#             grid[y_idx, x_idx + 1] += (1 - x_con) * y_con
#             grid[y_idx + 1, x_idx] += x_con * (1 - y_con) 
#             grid[y_idx + 1, x_idx + 1] += (1 - x_con) * (1 - y_con) 
            
#         if x_dist >= 0 and y_dist < 0:
#             grid[y_idx, x_idx] += x_con * y_con
#             grid[y_idx, x_idx + 1] += (1 - x_con) * y_con
#             grid[y_idx - 1, x_idx] += x_con * (1 - y_con) 
#             grid[y_idx - 1, x_idx + 1] += (1 - x_con) * (1 - y_con) 
            
#         if x_dist < 0 and y_dist >= 0:
#             grid[y_idx, x_idx] += x_con * y_con
#             grid[y_idx, x_idx - 1] += (1 - x_con) * y_con
#             grid[y_idx + 1, x_idx] += x_con  * (1 - y_con) 
#             grid[y_idx + 1, x_idx - 1] += (1 - x_con) * (1 - y_con) 
            
#         if x_dist < 0 and y_dist < 0:
#             grid[y_idx, x_idx] += x_con * y_con
#             grid[y_idx, x_idx - 1] += (1 - x_con) * y_con
#             grid[y_idx - 1, x_idx] += x_con * (1 - y_con) 
#             grid[y_idx - 1, x_idx - 1] += (1 - x_con) * (1 - y_con) 
    
#     return grid

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    train_dataset = PyWakeDataset("/home/frederikwr/Documents/PyWakeSimulations/res512_256_jensen/test",
                                transform=ToTensor(["WS_eff", "TI_eff"], ["WS", "TI"]))

    # train_loader = DataLoader(train_dataset, batch_size=2)

    print(next(iter(train_dataset)))