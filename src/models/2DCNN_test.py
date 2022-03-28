"""
SCRIPT FOR TESTING 2DCNN MODELS
"""
import time
import torch
import numpy as np
from datetime import datetime
from CNN import CNNmodel
from Training import ImbalancedDatasetUnderSampler
from Training import test_model
from Testing import *
from Data_maker_loader import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.empty_cache()

hyperparameter_defaults = dict(
    region="Junin",
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)],
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
    padding=[0, 0, 0, 0],
    size=36,
    dropout=0.1590181608038966,
    levels=8,
    batch_size=128,
    hidden_dim1=16,
    hidden_dim2=256,
    hidden_dim3=256,
    hidden_dim4=128,
    lr=0.004411168121921798,
    weight_decay=0,
    n_splits=5,
    # Set criteria for Early stopping
    AUC=True,
    BCE_Wloss=False,
    FNcond=False,
    n_epochs=40,
    patience=7,
    training_time=11.75,
    # Weight on BCE loss
    pos_weight=2,
    # set ratios of 0:1 labels in Train and Validation data sets
    train_times=4,
    test_times=4,
    # set parameters for the cost of the confusion matrix
    # weights on the False Negative Rate
    w=10,
    # Batch params
    stop_batch=1000000,
    print_batch=500,
    # Set training time period
    start_year=19,
    end_year=19,
    modeltype="2D",
)

region = hyperparameter_defaults["region"]
server = "/rds/user/jgcb3/hpc-work/forecasting/junin"
# server = '/rds/general/project/aandedemand/live/satellite/junin'
# server = "/rds/general/user/jgb116/home/satellite/satellite/junin"
# from where to load tensors for data
wherepath = server + "/data/" + region
# Where to get and save tif map
sourcepath = server + "/deforestation_forecasting/download_data/outputs/" + region
# sourcepath = server + "/data/rasters_junin"
# Where to get model checkpoint
modelpath = (
    server + "/models/Junin_models/2D/torch.nn.parallel.data_parallel.DataParallel"
)
bestmodel = "/torch.nn.parallel.data_parallel.DataParallel_8.12.21_11.33_51110015.pt"
checkpoint = modelpath + bestmodel
modelname = checkpoint.split("/", -1)[-1]
# Where to save Test_Roc
picspath = modelpath + "/pics"
checkpoint

# Set model, measurments and scenario parameters

# Set test time period
# predict 2017 images - 2018 labels
start_year = hyperparameter_defaults["start_year"]
end_year = hyperparameter_defaults["end_year"]

# set CNN model parameters
size = hyperparameter_defaults["size"]
DSM = False

# CHOOSE THE INPUT DIMENSIONS - No DSM is (2,8). With DSM is (3,8)
if DSM:
    input_dim = 11
else:
    input_dim = 10


hidden_dim = [
    hyperparameter_defaults["hidden_dim1"],
    hyperparameter_defaults["hidden_dim2"],
    hyperparameter_defaults["hidden_dim3"],
    hyperparameter_defaults["hidden_dim4"],
]
kernel_size = hyperparameter_defaults["kernel_size"]
stride = hyperparameter_defaults["stride"]
padding = hyperparameter_defaults["padding"]
dropout = hyperparameter_defaults["dropout"]
levels = hyperparameter_defaults["levels"]


# set ratios of 0:1 labels in Test data sets
train_times = hyperparameter_defaults["train_times"]
test_times = hyperparameter_defaults["test_times"]

# set parameters for the cost of the confusion matrix
w = hyperparameter_defaults["w"]  # weights on the False Negative Rate
perc = (100 * test_times) / (
    test_times + 1
)  # the percentile to for treshhold selection. Advisable to be 100*times/(times+1)

# Weight parameter for the weighted BCE loss
pos_weight = hyperparameter_defaults["pos_weight"]
# parameters for testing
stop_batch = hyperparameter_defaults["stop_batch"]
print_batch = hyperparameter_defaults["print_batch"]
batch_size = hyperparameter_defaults["batch_size"]

model = CNNmodel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    kernel_size=hyperparameter_defaults["kernel_size"],
    levels=[hyperparameter_defaults["levels"]],
    dropout=hyperparameter_defaults["dropout"],
    # start_year=config["start_year"],
    # end_year=config["end_year"],
    stride=hyperparameter_defaults["stride"],
    padding=hyperparameter_defaults["padding"],
)
model = torch.nn.DataParallel(model)

print("Checkpoint: " + checkpoint)
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint["model_state_dict"])

criterion = torch.nn.BCEWithLogitsLoss(
    reduction="mean", pos_weight=torch.tensor(pos_weight)
)

Data = with_DSM(
    size=int(size / 2),
    start_year=start_year,
    end_year=end_year,
    wherepath=wherepath,
    DSM=DSM,
    type=hyperparameter_defaults["modeltype"],
)
indeces = np.array(range(0, len(Data)))

test_sampler = ImbalancedDatasetUnderSampler(
    labels=Data.labels, indices=indeces, times=test_times
)
print(datetime.now())
print("Region: " + region)
print(modelname)
print(
    "percentage valid pixels in year 20%d with label 1: " % (end_year + 1),
    test_sampler.count[1] / sum(test_sampler.count),
)
print("Which correspond to %d number of 1 labeled pixels" % (test_sampler.count[1]))


# ## testing(model, Data, criterion, test_sampler, w, perc, batch_size, stop_batch, print_batch, name = None, path = None)

start = time.time()

outputs, targets, coordinates = testing(
    model=model,
    Data=Data,
    criterion=criterion,
    test_sampler=test_sampler,
    w=w,
    perc=perc,
    batch_size=batch_size,
    stop_batch=stop_batch,
    print_batch=print_batch,
    name=modelname,
    path=modelpath + "/CNN.CNNmodel",
    save=True,
)


# outputs, coordinates = forecasting(model = model,
#                                   Data = Data,
#                                   year = end_year,
#                                   batch_size = batch_size,
#                                   stop_batch = stop_batch,
#                                   print_batch = print_batch)

print("outputs")
print(outputs.shape)


print("coordinates")
print(coordinates.shape)

# for illustrative purposes create mock outputs and coordinates
from torch.distributions import Uniform

valid_pixels = torch.load(wherepath + "/pixels_cord_%d.pt" % (end_year))
m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
scores = m.sample((len(valid_pixels),))
scores = scores.squeeze(dim=1)

print("scores")
print(scores.shape)

print("valid_pixels")
print(valid_pixels.shape)


# # Heatmap

# sourcepath = '/rds/general/user/jgb116/home/repos/deforestation_forecasting/data/Hansen'


# heatmap(end_year = end_year,
#        outputs = outputs, # was scores, but this is just noise, right?
#        coordinates = valid_pixels,
#        sourcepath = sourcepath,
#        wherepath = wherepath,
#        name = modelname+"mock")

print("\n\nEND!Total time (in h):", (time.time() - start) / 3600)
