import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import yaml
from CustomLoss import WeightedLoss
from data_handler.dataset import GraphDataset
from models.GCN_CNN1 import GCN_CNN1
from torch_geometric.loader import DataLoader

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

config = ConfigLoader.from_yaml("/home/frederikwr/Dropbox/Master_Thesis_public/machine_learning/config.yaml")
train_dataset = GraphDataset(config.io_settings.train_dataset_path, ["WS_eff"]) 
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=config.run_settings.num_workers_train)

model = GCN_CNN1(5, 16, 256, 1024, 1)
optimizer = optim.Adam(model.parameters(), lr=float(config.hyperparameters.start_lr))
criterion = WeightedLoss(loss_type="mse", split_ratio=[25, 75],weights=[0.1, 0.9])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
for epoch in range(num_epochs):

    model.train() 
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        graph, flow_map = data["graph"], data["flow_map"]


        graph, flow_map = graph.to(device), flow_map.to(device)
        model.to(device)

        optimizer.zero_grad()
        outputs = model(graph)
        loss = criterion(outputs, flow_map)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Training')

model.to("cpu")
output = model(next(iter(train_loader))["graph"]).detach().numpy()
print(output.shape)

fig, ax = plt.subplots()
ax.contourf(output[0,0,:,:])
plt.show()