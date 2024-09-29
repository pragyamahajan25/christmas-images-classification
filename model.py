import torch
import torch.nn as nn
from torchvision.models import resnet50
import os

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.resnet = resnet50()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 8) 
        
        
    def forward(self, x):
        x = self.resnet(x)
        return x
       
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model.pkl')

