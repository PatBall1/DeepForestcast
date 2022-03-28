import torch
import numpy as np
from spp_layer import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNmodel(torch.nn.Module):

    def __init__(     
        self, input_dim=10, 
        hidden_dim=[128,64,64,32],
        kernel_size = [(5,5),(5,5),(3,3),(3,3)],
        stride = [(2,2),(1,1),(1,1),(1,1)],
        padding =[0,0,0,0],
        dropout = 0.2,
        levels=[13]
        ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.levels = levels
                                                  
        self.conv = torch.nn.Sequential(        
            torch.nn.Conv2d(input_dim,hidden_dim[0],kernel_size[0],stride = stride[0], padding=padding[0]), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            
            torch.nn.Conv2d(hidden_dim[0],hidden_dim[1],kernel_size[1],stride[1],padding=padding[1]), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[1]),
     
            torch.nn.Conv2d(hidden_dim[1],hidden_dim[2],kernel_size[2],stride[2],padding=padding[2]), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            
            torch.nn.Conv2d(hidden_dim[2],hidden_dim[3],kernel_size[3],stride[3],padding=padding[3]), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[3]))
        
        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[-1]*i*i        
        self.ln = torch.nn.Sequential( 
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),           
            torch.nn.Linear(100, 1))

        self.sig = torch.nn.Sigmoid()
        
    def forward(self , x, sigmoid = True):
        x = self.conv(x)
        x = spp_layer(x, self.levels)
        x= self.ln(x)
        if sigmoid: 
            x = self.sig(x)            
        return x.flatten()


if __name__=='__main__':
    model = CNNmodel()
    in_size = 45
    x = torch.rand([2,10,in_size,in_size])
    y = model.forward(x)
    print(y.shape)

