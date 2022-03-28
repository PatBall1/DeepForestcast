import torch
from Data_maker_loader import *
from Testing import *
from torch.utils.data import DataLoader
from CNN import *

server = '/rdsgpfs/general/user/kpp15/home/Hansen'
wherepath = server +'/data/raster/tensors'
sourcepath = server + '/data/raster/MadreTiff'
modelpath = server + "/deforestation_forecasting/models"

#set CNN model parameeters
checkpoint = modelpath + "/CNN.CNNmodel/CNN.CNNmodel_3.7.19_23.47_315834[9].pbs.pt"
modelname = checkpoint.split("/",-1)[-1]
end_year = 18
size = 45
DSM = True
input_dim=11 
hidden_dim=[128,64,64,32]
kernel_size = [(5,5),(5,5),(3,3),(3,3)]
stride = [(2,2),(1,1),(1,1),(1,1)]
padding =[0,0,0,0]
dropout = 0.2
levels=[13]

batch_size = 100
print_batch = 200
stop_batch = None


if __name__=="__main__":
    
    Data = load_CNNdata_forecast(size = int(size/2), year = end_year, path = wherepath)
    
    outputs, coordinates = forecasting(model = model,
                                   Data = Data,
                                   year = end_year,
                                   batch_size = batch_size,
                                   stop_batch = stop_batch,
                                   print_batch = print_batch)
    
    heatmap(end_year = end_year,
        outputs = outputs,
        coordinates = coordinates,
        sourcepath = sourcepath,
        wherepath = wherepath,
        name = modelname+"_forecast_from")