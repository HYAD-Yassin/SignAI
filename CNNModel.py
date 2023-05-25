import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from CustomImageDataset import CustomImageDataset

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 100 * 100, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Specify the root directory where your "DataBase" folder is located
root_dir = "DataBase/"

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Instantiate the custom dataset
custom_dataset = CustomImageDataset(root_dir, transform=transform)
print("Custom dataset created")

# Create a data loader
batch_size = 4
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Specify the number of classes in your dataset
num_classes = len(custom_dataset.classes)

# Instantiate the CNN model
model = CNNModel()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)  


        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

print("Training completed!")


# Save the trained model
model_path = "model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")