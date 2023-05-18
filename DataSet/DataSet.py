import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
            self.images.extend(class_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = os.path.basename(os.path.dirname(img_path))  # Extract the label from the parent folder name

        return image, label

# Specify the root directory where your "DataBase" folder is located
root_dir = "DataBase/"

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

# Instantiate the custom dataset
custom_dataset = CustomImageDataset(root_dir, transform=transform)

# Create a data loader
batch_size = 4
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)



# fetch the data loader
#for images, labels in data_loader:
    # Access a batch of images and labels
    #print(f"Images shape: {images.shape}, Labels: {labels}")



# Specify the character you want to test
test_character = 'C'
to_pil = transforms.ToPILImage()


# Iterate through the dataset and find images of the test character
for image, label in custom_dataset:
    if label == test_character:
        # Process or display the image as needed
        print(f"Label: {label}")
         # Convert the tensor to PIL Image
        pil_image = to_pil(image)
        # Display the image
        pil_image.show()
        break  # Break out of the loop after finding the first image
