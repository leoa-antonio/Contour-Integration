import torch
import torch.nn as nn

class ContourCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=64, out_features=1)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x
    
if __name__ == "__main__":
    model = ContourCNN()

    x = torch.randn(4, 1, 128, 128)
    output = model(x)

    assert output.shape == (4, 1)
    print("CNN forward pass successful!")