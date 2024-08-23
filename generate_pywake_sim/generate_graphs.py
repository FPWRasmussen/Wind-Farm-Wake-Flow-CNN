
import os

import torch
import xarray as xr
import yaml
from torch_geometric.data import Data
from torch_geometric.transforms import Delaunay, FaceToEdge
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import add_self_loops

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


def generate_graphs(config):
    # directory_path = config.pywake_results_settings.dataset_path
    # train_path = os.path.join(directory_path, "train")
    # test_path = os.path.join(directory_path, "test")
    # train_folders = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
    # test_folders = [name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))]
    # folders = train_folders + test_folders
    directory_path = config.pywake_results_settings.dataset_path
    train_path = os.path.join(directory_path, "train")
    test_path = os.path.join(directory_path, "test")
    train_folders = [os.path.join(train_path, name) for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
    test_folders = [os.path.join(test_path, name) for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))]
    folders = train_folders + test_folders

    for folder in folders:
        try:
            sim_res_path = os.path.join(folder, "sim_res.nc")
            sim_res = xr.load_dataset(sim_res_path)
        except:
            continue

        pos = torch.tensor(sim_res[["x", "y"]].to_array().values.T)
        x = torch.tensor(sim_res[["WS_eff", "TI_eff", "CT"]].to_array().values.squeeze().T)
        globals = torch.tensor(sim_res[["WS", "TI"]].to_array().values.squeeze().T)
        # if x.dim() == 1:
        #     x = x.unsqueeze(1)  # Add a second dimension
            
        if config.graph_generation.graph_type == "delaunay":
            if len(pos) > 2:
                data = Data(pos=pos, x = x, globals = globals)
                d = Delaunay()
                e = FaceToEdge()
                data = e(d(data))
            else:
                adj = torch.ones(pos.shape[0], pos.shape[0])
                data = Data(pos=pos, x = x, edge_index=dense_to_sparse(adj.fill_diagonal_(0))[0])
        elif config.graph_generation.graph_type == "fully_connected":
            adj = torch.ones(pos.shape[0], pos.shape[0])
            data = Data(pos=pos, x = x, globals = globals, edge_index=dense_to_sparse(adj.fill_diagonal_(0))[0])

        else:
            raise NotImplementedError(f"The graph type '{config.graph_generation.graph_type}' is not a valid type.")
        
        if data.edge_index.numel() == 0:
            data.x = data.x.unsqueeze(0)
            data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(1))
            
        torch.save(data, os.path.join(folder, "graph.pt"))
        

if __name__ == "__main__":
    config = ConfigLoader.from_yaml('/home/frederikwr/Dropbox/Master_Thesis_public/generate_pywake_sim/config.yaml')
    generate_graphs(config)
    print("Done!")