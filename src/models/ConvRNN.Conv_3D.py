import os
import time
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Data_maker_loader import *
from ConvRNN import *
from Training import *
import scikitplot
from scipy.special import expit as Sigmoid

server = '/rds/general/project/aandedemand/live/satellite/junin'

# From where to load tensors for data
wherepath = server + '/data/tensors'

#Where to save model checkpoint
modelpath = server + "/models"

#Where to save images tracking training process
picspath = server + "/models/pics"

#Where to save results for models performance of each job on train test and val data set 
file = server + "/models/grid_summary/ConvRNN.Conv_3D.txt"


if __name__=="__main__":

    start_time = time.time()
    
    # Set training time period
    start_year = 14
    end_year = 17

    #set image parameters
    size = 45
    #set model parameters for 3D_CNN
    input_dim= (2,8)
    hidden_dim=(16,32,32)
    kernel_size=((5,5),(2,5,5),(5,5))
    levels=(10,)
    dropout = 0.3

    #set learning parameters
    train_times = 4
    test_times = 4
    
    #Set criteria for early stopping
    AUC = False
    BCE_Wloss = False
    FNcond = True
    # Set parameters for the cost of the confusion matrix
    w = 10
    perc = 80

    
    # Weight parameter for the weighted BCE loss
    pos_weight = 3
    
    # Adam optimiser parameters
    lr = 0.0001 
    weight_decay = 0    
    
    # Early stopping parameters
    n_epochs = 7
    n_splits = 5
    patience = 3

    training_time = 15
    stop_batch = None
    print_batch = 100
    batch_size = 10

    # To exploit different parameters in parallel
    if 'PBS_ARRAY_INDEX' in os.environ:
        job = int(os.environ['PBS_ARRAY_INDEX'])
    else:
        job = 3
    if 'PBS_JOBID' in os.environ:
        job_id  = str(os.environ['PBS_JOBID'])
    else:
        job_id = '1'
        
    if job == 2:
        kernel_size=((3,3),(2,3,3),(3,3))
        print("\nJob:",job, "parm selected:", kernel_size)

    if job == 3:
        hidden_dim=(16,32,32)
        kernel_size=((5,5),(2,5,5),(5,5))
        levels=(10,)
        training_time = 30
        stop_batch = 10
        print_batch = 1
        batch_size = 128


    if job == 4:
        levels=(10,5)
        print("\nJob:",job, "parm selected:", levels)

    if job == 6:
        size = 35 
        print("\nJob:",job, "parm selected:", size )

    if job == 7:
        size = 55
        print("\nJob:",job, "parm selected:", size )

    if job == 8:
        w = 5
        print("\nJob:",job, "parm selected:", w )

    if job == 9:
        pos_weight = 10
        print("\nJob:",job, "parm selected: ", pos_weight)

    if job == 10:
        pos_weight = 15
        print("\nJob:",job, "parm selected: ", pos_weight)

    if job == 11:
        weight_decay = 0.6
        print("\nJob:",job, "parm selected: ", weight_decay)

    if job == 12:
        lr = 0.001 
        print("\nJob:",job, "parm selected:", lr)


    Data = load_RNNdata(size = int(np.floor(size/2))  , start_year = start_year , end_year = end_year, path = wherepath)

    if not (os.path.isfile(wherepath+"/"+"Train3Da_idx%d.npy"%(end_year)) & os.path.isfile(wherepath+"/"+"Test3Da_idx%d.npy"%(end_year))):
        print("Creating indexes split")
        train_idx, test_idx = train_test_split(np.arange(len(Data.labels)),
                                                test_size=0.2, random_state=42, shuffle=True, stratify=Data.labels)
        np.save(wherepath+"/"+"Train3Da_idx%d.npy"%(end_year),train_idx)
        np.save(wherepath+"/"+"Test3Da_idx%d.npy"%(end_year),test_idx)
    else:
        train_idx = np.load(wherepath+"/"+"Train3Da_idx%d.npy"%(end_year))
        test_idx = np.load(wherepath+"/"+"Test3Da_idx%d.npy"%(end_year)) 

    train_sampler = ImbalancedDatasetUnderSampler(labels = Data.labels, indices=train_idx, times = train_times)

    test_sampler = ImbalancedDatasetUnderSampler(labels = Data.labels, indices=test_idx, times = test_times)
    test_sampler = SubsetRandomSampler(test_sampler.classIndexes_unsampled)
    test_loader = DataLoader(Data, sampler = test_sampler, batch_size=batch_size)

    model = Conv_3D(
                input_dim = input_dim,
                hidden_dim = hidden_dim,
                kernel_size= kernel_size,
                levels=levels,
                dropout = dropout)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight = torch.tensor(pos_weight) )
    optimiser = torch.optim.Adam(params=model.parameters(), lr = lr , weight_decay = weight_decay)

    #Print Summary of the training parameters
    # =========================================================================    
    print("Model:",str(type(model))[8:-2],"Period 20%d-20%d -> 20%d"%(start_year,end_year,end_year+1))   
    print("Job: ",job)   
    print("\nHyperparameters: ")
    print("\tImage size: %d"%(size))
    print("\tHidden dim: ",hidden_dim)
    print("\tTrain and Test ratios of 0:1 labels: 1:%d ; 1:%d "%(train_times, test_times))
    print("\tADAM optimizer parameters: lr=%.7f, weight decay=%.2f, batch size=%d"%(lr,weight_decay,batch_size))
    print("\tBCEWithLogitsLoss pos_weights = %.2f"%(pos_weight))
    print("\tn_epochs = %d with patience of %d epochs"%(n_epochs,patience))
    print("\tCross Validation with n_splits = %d "%(n_splits))
    print("\tIf to use BCEWithLogitsLoss as an early stop criterion :",((not AUC)&(not FNcond)))
    print("\tIf to use AUC as an early stop criterion :",AUC)
    print("\tIf to use cost = FP+w*FN / TP+FP+w*FN+TN as an early stop criterion")
    print("\twith w = %d and treshhold = the %d percentile of the output"%(w,perc),FNcond)
    print("\nModel: \n",model)
    print("\nCriterion: \n",criterion)
    print("\nOptimiser: \n",optimiser)

    model, train_loss, valid_loss, AUCs_train, AUCs_val ,costs_train, costs_val, name = train_model(Data = Data,
                model = model,
                sampler = train_sampler,
                criterion = criterion,
                optimiser = optimiser,
                patience = patience,
                n_epochs = n_epochs,
                n_splits = n_splits,                                                                                    
                batch_size = batch_size,
                stop_batch = stop_batch,
                print_batch = print_batch,
                training_time = training_time,
                w = w,
                perc = perc,
                FNcond = FNcond,
                AUC = AUC,
                job = job,
                path = modelpath)

    visualize(train = train_loss, valid = valid_loss, name = "BCEloss",
              modelname = name, best = "min", path = picspath)
    visualize(train = AUCs_train, valid = AUCs_val, name = "AUC",
              modelname = name, best = "max", path = picspath)
    visualize(train = costs_train, valid = costs_val, name = "Cost",
              modelname = name, best = "min", path = picspath)

    # Performance on test   
    # =============================================================================
    # initialize lists to monitor test loss and accuracy
    print("\nPerformance on test")
    test_losses = []
    output_total = []
    target_total = []
    model.eval() # prep model for evaluation
    require_sigmoid = isinstance(criterion,torch.nn.modules.loss.BCEWithLogitsLoss)
    #Need 'cor' here?
    for batch, (data, target, cor) in enumerate(test_loader):
        # Load onto GPU
        if type(data) == type([]):              
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:   
            data = data.to(device)
        target = target.to(device)
        cor = cor.to(device)
                
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data, sigmoid = not require_sigmoid)        
        # calculate the loss
        loss = criterion(output, target.float())
        test_losses.append(loss.item())
        output_total.append(list(output.data))
        target_total.append(list(target.data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break          
        # =====================================================================
    output_total = sum(output_total, [])
    target_total = sum(target_total, [])
    test_losses = np.average(test_losses)
    print('\tTest Loss: ',test_losses)
    test_AUC , test_cost = AUC_CM(target_total, output_total, w, perc, sigmoid = require_sigmoid)
    # ADDED THIS IF AS IT MATCHES Training.py - no sure if it's right!
    if require_sigmoid:
        output_total = np.array(Sigmoid(output_total))
    probas_per_class = np.stack((1 - output_total,output_total),axis = 1)
    roc = scikitplot.metrics.plot_roc(np.array(target_total),probas_per_class)
    path = os.path.join(picspath,name+"_ROC.png")
    roc.get_figure().savefig(path, bbox_inches='tight')

    print("\nTotal time needed(h): ",(time.time() - start_time)/3600)

    best_epoch = costs_val.index(min(costs_val))
    summary = f'| Job:{job:d}'+' | Model:'+name+f' | Best epoch:{best_epoch+1:d} |'
    Train = f'| Train loss:{train_loss[best_epoch]:.3f} | Train AUC:{AUCs_train[best_epoch]:.3f} | Train cost:{costs_train[best_epoch]:.3f} |'
    Val = f'|   Val loss:{valid_loss[best_epoch]:.3f} |   Val AUC:{AUCs_val[best_epoch]:.3f} |   Val cost:{costs_val[best_epoch]:.3f} |' 
    Test = f'|  Test loss:{test_losses:.3f} |  Test AUC:{test_AUC:.3f} |  Test cost:{test_cost:.3f} |'
    print(summary)
    print(Train)
    print(Val)
    print(Test)
    file = open(file,"a") 
    file.write("\n\n")
    file.write(summary)
    file.write("\n")
    file.write(Train) 
    file.write("\n")
    file.write(Val)
    file.write("\n")
    file.write(Test) 
    file.write("\n")
    file.close() 