import torch
import numpy as np
from CNN import *
from Training import ImbalancedDatasetUnderSampler
from Testing import *
from Data_maker_loader import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
import time

server = '/rdsgpfs/general/user/kpp15/home/Hansen'
#from where to load tensors for data
wherepath = server + '/data/raster/tensors'
#Where to get and save tif map 
sourcepath = server + '/data/raster/MadreTiff'
#Where to get model checkpoint
modelpath = server + "/deforestation_forecasting/models"
checkpoint = modelpath + "/CNN.CNNmodel/CNN.CNNmodel_3.7.19_23.47_315834[9].pbs.pt"
modelname = checkpoint.split("/",-1)[-1]
#Where to save Test_Roc 
picspath = server + "/deforestation_forecasting/models/pics"

if __name__=="__main__":
        
    start = time.time()
    #Set test time period
    # predict 2017images - 2018lanels 
    start_year = 17
    end_year = 17

    #set CNN model parameeters
    size = 45
    DSM = True
    input_dim=11 
    hidden_dim=[128,64,64,32]
    kernel_size = [(5,5),(5,5),(3,3),(3,3)]
    stride = [(2,2),(1,1),(1,1),(1,1)]
    padding =[0,0,0,0]
    dropout = 0.2
    levels=[13]

    #set ratios of 0:1 labels in Test data sets
    test_times = 4

    # set parameters for the cost of the confussion matrix
    w = 10 # weights on the False Negative Rate
    perc = (100*test_times) / (test_times+1) # the percentile to for treshhold selection. Advisable to be 100*times/(times+1)

    # Weight parameter for the weighted BCE loss
    pos_weight = 3

    #parameters for testing
    stop_batch = None
    print_batch = 200
    batch_size = 200
    
  
    model = CNNmodel(
                input_dim = input_dim,
                hidden_dim = hidden_dim,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dropout = dropout,
                levels = levels)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight = torch.tensor(pos_weight))

    
    Data =  with_DSM(size = int(size/2), start_year = start_year, end_year = end_year, wherepath = wherepath , DSM = DSM)
    indeces = np.array(range(0,len(Data)))
    test_sampler = ImbalancedDatasetUnderSampler(labels = Data.labels, indices= indeces , times = test_times)
    
    print(modelname)
    print("\n\n")
    print("percentage valid pixels in year 20%d with label 1: "%(end_year +1 ), test_sampler.count[1]/sum(test_sampler.count))
    print("Which correspond to %d number of 1 labaled pixels."%(test_sampler.count[1])) 
    print("\nHyperparameters: ")
    print("\tImage size: %d"%(size))
    print("\tTest ratio of 0:1 labels: 1:%d"%(test_times))
    print("\tBCEWithLogitsLoss pos_weights = %.2f"%(pos_weight))
    print("\tWeight w = %d and treshhold = the %d percentile of the output"%(w,perc))
    print("\nModel: \n",model)
    print("\nCriterion: \n",criterion)

    outputs, targets, coordinates = testing(model = model,
                                               Data = Data,
                                               criterion = criterion,
                                               test_sampler = test_sampler,
                                               w = w, 
                                               perc = perc,
                                               batch_size = batch_size,
                                               stop_batch = stop_batch,
                                               print_batch = print_batch,
                                               name = modelname,
                                               path = modelpath + "/CNN.CNNmodel",
                                               save = True)

    heatmap(end_year = end_year,
            outputs = outputs,
            coordinates = coordinates,
            sourcepath = sourcepath,
            wherepath = wherepath,
            name = modelname+"mock")
    
    print("\n\nEND!Total time (in h):",(time.time()-start)/3600)