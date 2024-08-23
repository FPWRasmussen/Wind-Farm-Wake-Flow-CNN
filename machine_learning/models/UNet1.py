import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, num_extra_features):
        super(UNet1, self).__init__()
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
        
        # Latent space
        self.fc_encode = nn.Linear(256 + num_extra_features, 256)
        
        # Decoder
        self.decode1 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.decode2 = nn.ConvTranspose2d(64+128, 64, 3, 1, 1)
        self.decode3 = nn.ConvTranspose2d(64, 32, 3, 1, 1)
        self.decode4 = nn.ConvTranspose2d(32+64, 32, 3, 1, 1)
        self.decode5 = nn.ConvTranspose2d(32, 16, 5, 1, 2)
        self.decode6 = nn.ConvTranspose2d(16+32, 16, 5, 2, 2, output_padding=1)
        self.decode7 = nn.ConvTranspose2d(16, 8, 5, 1, 2)
        self.decode8 = nn.ConvTranspose2d(8+16, 8, 5, 2, 2, output_padding=1)
        self.decode9 = nn.ConvTranspose2d(8, 4, 5, 1, 2)
        self.decode10 = nn.ConvTranspose2d(4+8, 4, 5, 2, 2, output_padding=1)
        self.decode11 = nn.ConvTranspose2d(4, out_channels, 5, 1, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, extra_vars):
        # Encoder
        enc1 = F.relu(self.encode1(x))
        enc2 = F.relu(self.encode2(enc1))
        enc3 = F.relu(self.encode3(enc2))
        x, indices1 = F.max_pool2d(enc3, 2, return_indices=True)
        enc4 = F.relu(self.encode4(x))
        enc5 = F.relu(self.encode5(enc4))
        x, indices2 = F.max_pool2d(enc5, 2, return_indices=True)
        enc6 = F.relu(self.encode6(x))
        enc7 = F.relu(self.encode7(enc6))
        x, indices3 = F.max_pool2d(enc7, 2, return_indices=True)
        enc8 = F.relu(self.encode8(x))
        enc9 = F.relu(self.encode9(enc8))
        x, indices4 = F.max_pool2d(enc9, 2, return_indices=True)
        enc10 = F.relu(self.encode10(x))
        enc11 = F.relu(self.encode11(enc10))
        x_mark, indices5 = F.max_pool2d(enc11, 2, return_indices=True)
        

        x = x_mark.reshape(x_mark.size(0), -1)
        x = torch.cat((x, extra_vars), dim=1)
        x = self.dropout(F.relu(self.fc_encode(x)))
        x = x.reshape(x_mark.shape)
        
        # Decoder
        x = F.relu(self.decode1(F.max_unpool2d(x, indices5, 2)))
        x = torch.cat((x, enc11), dim=1)
        x = F.relu(self.decode2(x))
        x = F.relu(self.decode3(F.max_unpool2d(x, indices4, 2)))
        x = torch.cat((x, enc9), dim=1)
        x = F.relu(self.decode4(x))
        x = F.relu(self.decode5(F.max_unpool2d(x, indices3, 2)))
        x = torch.cat((x, enc7), dim=1)
        x = F.relu(self.decode6(x))
        x = F.relu(self.decode7(F.max_unpool2d(x, indices2, 2)))
        x = torch.cat((x, enc5), dim=1)
        x = F.relu(self.decode8(x))
        x = F.relu(self.decode9(F.max_unpool2d(x, indices1, 2)))
        x = torch.cat((x, enc3), dim=1)
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
    model = UNet(in_channels=num_channels, out_channels=num_classes, num_extra_features=num_extra_features)

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