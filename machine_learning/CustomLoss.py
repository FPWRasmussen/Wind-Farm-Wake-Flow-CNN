import torch
import torch.nn as nn

class WeightedLoss(nn.Module):
    def __init__(self, loss_type, split_ratio, weights):
        super(WeightedLoss, self).__init__()
        self.split_ratio = split_ratio
        self.weights = weights
        if loss_type == "mse":
            self.loss = nn.MSELoss(reduction='none')
        elif loss_type == "l1":
            self.loss = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        
        split_pct = self.split_ratio[0]/(sum(self.split_ratio))
        split_col = int(output.shape[-1]*split_pct)
        
        loss = self.loss(output, target)
        loss[:, :, :, split_col:] *= self.weights[0]
        loss[:, :, :, :split_col] *= self.weights[1]
        loss = torch.mean(loss)
        return loss
    
    
if __name__ == "__main__":
    class ExampleModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(ExampleModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Example usage
    input_size = 10
    output_size = 256
    batch_size = 32

    model = ExampleModel(input_size, output_size)

    # Create dummy input data
    input = torch.randn(batch_size, 1, input_size, input_size)
    input[:,:,:,input_size//2:] = 0

    # Create dummy target data
    target_data = torch.randn(batch_size, 1, input_size, output_size)

    # Create weight tensor
    split_ratio = [25, 75]
    weights = [0.1, 0.9]

    # Create loss function instance
    loss_fn = WeightedLoss("mse", split_ratio, weights)

    # Forward pass
    output = model(input)

    # Compute loss
    loss = loss_fn(output, target_data)

    loss.backward()

    print(loss)