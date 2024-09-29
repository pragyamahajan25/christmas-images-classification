import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import natsort

# Class 
class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.training:
            self.dataset = ImageFolder(root=self.path, transform=self.transform)
        else:
            self.image_files = os.listdir(self.path)
            
    def __len__(self):
        if self.training:
            return len(self.dataset)
        else:
            return len(self.image_files)

        
    def __getitem__(self, index):
          if self.training:
            return self.dataset[index]
          else:
            image_name = self.image_files[index]
            image_path = os.path.join(self.path, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, int(image_name.split('.')[0])
        
