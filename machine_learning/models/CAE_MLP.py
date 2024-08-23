import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_extra_features):
        super(CAE_MLP, self).__init__()
        # Encoder
        self.encode1 = nn.Conv2d(in_channels, 4, 5, 1, 2)
        self.encode2 = nn.Conv2d(4, 4, 5, 2, 2)
        self.encode3 = nn.Conv2d(4, 8, 5, 1, 2)
        self.encode4 = nn.Conv2d(8, 8, 5, 2, 2)
        self.encode5 = nn.Conv2d(8, 16, 5, 1, 2)
        self.encode6 = nn.Conv2d(16, 16, 5, 2, 2)
        self.encode7 = nn.Conv2d(16, 32, 5, 1, 2)
        self.encode8 = nn.Conv2d(32, 32, 3, 1, 1)
        self.encode9 = nn.Conv2d(32, 64, 3, 1, 1)
        self.encode10 = nn.Conv2d(64, 64, 3, 1, 1)
        self.encode11 = nn.Conv2d(64, 128, 3, 1, 1)
        
        self.fc_encode = nn.Sequential(
            nn.Linear(256+ num_extra_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        
        # Decoder
        self.decode1 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.decode2 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.decode3 = nn.ConvTranspose2d(64, 32, 3, 1, 1)
        self.decode4 = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        self.decode5 = nn.ConvTranspose2d(32, 16, 5, 1, 2)
        self.decode6 = nn.ConvTranspose2d(16, 16, 5, 2, 2, output_padding=1)
        self.decode7 = nn.ConvTranspose2d(16, 8, 5, 1, 2)
        self.decode8 = nn.ConvTranspose2d(8, 8, 5, 2, 2, output_padding=1)
        self.decode9 = nn.ConvTranspose2d(8, 4, 5, 1, 2)
        self.decode10 = nn.ConvTranspose2d(4, 4, 5, 2, 2, output_padding=1)
        self.decode11 = nn.ConvTranspose2d(4, out_channels, 5, 1, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # MaxUnpool layers
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.unpool5 = nn.MaxUnpool2d(2, 2)

    def forward(self, x, extra_vars):
        # Encoder
        x = F.relu(self.encode1(x))
        x = F.relu(self.encode2(x))
        x = F.relu(self.encode3(x))
        x, indices1 = F.max_pool2d(x, 2, return_indices=True)
        x = F.relu(self.encode4(x))
        x = F.relu(self.encode5(x))
        x, indices2 = F.max_pool2d(x, 2, return_indices=True)
        x = F.relu(self.encode6(x))
        x = F.relu(self.encode7(x))
        x, indices3 = F.max_pool2d(x, 2, return_indices=True)
        x = F.relu(self.encode8(x))
        x = F.relu(self.encode9(x))
        x, indices4 = F.max_pool2d(x, 2, return_indices=True)
        x = F.relu(self.encode10(x))
        x = F.relu(self.encode11(x))
        x_mark, indices5 = F.max_pool2d(x, 2, return_indices=True)
        
        # Flatten and concatenate with extra variables
        x = x_mark.reshape(x_mark.size(0), -1)
        x = torch.cat((x, extra_vars), dim=1)
        
        # Latent space
        x = self.dropout(F.relu(self.fc_encode(x)))
        
        # Reshape for decoder
        x = x.reshape(x_mark.shape)
        
        # Decoder
        x = self.unpool1(x, indices5)
        x = F.relu(self.decode1(x))
        x = F.relu(self.decode2(x))
        x = self.unpool2(x, indices4)
        x = F.relu(self.decode3(x))
        x = F.relu(self.decode4(x))
        x = self.unpool3(x, indices3)
        x = F.relu(self.decode5(x))
        x = F.relu(self.decode6(x))
        x = self.unpool4(x, indices2)
        x = F.relu(self.decode7(x))
        x = F.relu(self.decode8(x))
        x = self.unpool5(x, indices1)
        x = F.relu(self.decode9(x))
        x = F.relu(self.decode10(x))
        x = self.decode11(x)  # No activation on the final layer
        
        return x
    
if __name__ == "__main__":
    # Generate random data
    batch_size = 3
    num_channels = 3
    image_size = 256
    num_classes = 10
    num_extra_features = 4

    inputs = torch.randn(batch_size, num_channels, image_size*2, image_size)
    outputs = torch.randn(batch_size, num_classes, image_size*2, image_size)
    labels = torch.randn(batch_size,num_extra_features)

    # Instantiate the model
    model = CAE_MLP(in_channels=num_channels, out_channels=num_classes, num_extra_features=num_extra_features)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of parameters: {}".format(total_params))
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        model_outputs = model(inputs, labels)
        loss = criterion(model_outputs, outputs)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")