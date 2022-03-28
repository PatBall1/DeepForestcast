import torch
from Data_maker_loader import *
from Testing import *
from torch.utils.data import DataLoader
from CNN import *

#server = '/rdsgpfs/general/user/kpp15/home/Hansen'
#wherepath = server +'/data/raster/tensors'
#sourcepath = server + '/data/raster/MadreTiff'
#modelpath = server + "/deforestation_forecasting/models"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Device: " + torch.cuda.get_device_name(0))
    print ("Device count: " + str(torch.cuda.device_count()))
else:
    print("torch.cuda.is_available == False")

# Should the model be parallelised for multiple GPU use?
parallelise = True
print("Parallelised for multiple GPUs:", parallelise)


server = '/rds/general/project/aandedemand/live/satellite/junin'
wherepath = server + '/data/tensors'
sourcepath = '/rds/general/user/jgb116/home/repos/deforestation_forecasting/data/Hansen'
modelpath = server + "/models"
checkpoint = modelpath + "/torch.nn.parallel.data_parallel.DataParallel/torch.nn.parallel.data_parallel.DataParallel_17.9.19_7.53_547402[10].pbs.pt"
modelname = checkpoint.split("/",-1)[-1]
#Where to save Test_Roc 
picspath = server + "/deforestation_forecasting/models/pics"

#set CNN model parameeters
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

Data = load_CNNdata_forecast(size = int(size/2), year = end_year, path = wherepath)
size = 45
DSM = True
input_dim=11 
hidden_dim=[128,64,64,32]
kernel_size = [(5,5),(5,5),(3,3),(3,3)]
stride = [(2,2),(1,1),(1,1),(1,1)]
padding =[0,0,0,0]
dropout = 0.2
levels=[13]

model = CNNmodel(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dropout = dropout,
            levels = levels)
if parallelise:
    model = torch.nn.DataParallel(model)
checkpoint = torch.load(checkpoint)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

end_year = 18
batch_size = 1024
print_batch = 1000
stop_batch = 5000 #None
outputs, coordinates = forecasting(model = model,
                               Data = Data,
                               year = end_year,
                               batch_size = batch_size,
                               stop_batch = stop_batch,
                               print_batch = print_batch)
np.savetxt("/rds/general/user/jgb116/home/repos/deforestation_forecasting/data/Hansen/heatmaps/" + modelname + "_forecast_from_outputs.csv", 
            np.c_[coordinates.numpy(),outputs.numpy()],delimiter=",")
print("output saved in " + "/rds/general/user/jgb116/home/repos/deforestation_forecasting/data/Hansen/heatmaps/" + modelname + "_forecast_from_outputs.csv") 
heatmap(end_year = end_year,
    outputs = outputs,
    coordinates = coordinates,
    sourcepath = sourcepath,
    wherepath = wherepath,
    name = modelname+"_forecast_from")
print("files saved in")
print(modelname + "_forecast_from")


