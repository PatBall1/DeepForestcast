import os
from matplotlib import cm
import torch
import numpy as np
import time
import wandb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from torch.utils.data import DataLoader
from CNN import CNNmodel
from Training import test_model, train_model
from Training import visualize, write_report, ImbalancedDatasetUnderSampler
from Data_maker_loader import with_DSM

server = "/rds/user/jgcb3/hpc-work/forecasting/junin/"

hyperparameter_defaults = dict(
    region="Junin",
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)],
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
    padding=[0, 0, 0, 0],
    size=5,
    dropout=0.1590181608038966,
    levels=8,
    batch_size=1024,
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
    stop_batch=None,
    print_batch=500,
    # Set training time period
    start_year=14,
    end_year=18,
    modeltype="2D",
)

config = hyperparameter_defaults
# Years


# WHERE TO IMPORT DATA FROM
wherepath = server + "/data/" + config["region"]
savepath = server + "/data/" + config["region"] + "/out"
if not os.path.exists(savepath):
    os.makedirs(savepath)

# WHERE TO SAVE MODEL CHECKPOINT
modelpath = server + "/models/" + config["region"] + "_models/3D"
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

# WHERE TO SAVE IMAGES TRACKING TRAINING PROCESS
picspath = server + "/models/" + config["region"] + "_models/3D/pics"
if not os.path.exists(picspath):
    os.makedirs(picspath)

# WHERE TO SAVE MODEL PERFORMANCE OF EACH JOB FOR TRAIN, VAL AND TEST DATA
file = (
    server
    + "/models/"
    + config["region"]
    + "_models/3D/grid_summary/ConvRNN.Conv_3D.txt"
)
if not os.path.exists(os.path.dirname(file)):
    os.makedirs(os.path.dirname(file))

DSM = False
Data = with_DSM(
    size=int(config["size"] / 2),
    start_year=config["start_year"],
    end_year=config["end_year"],
    wherepath=wherepath,
    DSM=DSM,
    type=config["modeltype"],
)
Data[0][0].flatten().shape

train_idx = np.load(wherepath + "/" + "Train2D_idx%d.npy" % (config["end_year"]))
test_idx = np.load(wherepath + "/" + "Test2D_idx%d.npy" % (config["end_year"]))

# Set train and test samplers

# c = DataLoader(Data)
train_sampler = ImbalancedDatasetUnderSampler(
    labels=Data.labels, indices=train_idx, times=config["train_times"]
)
tr_idx = train_sampler.classIndexes_unsampled
test_sampler = ImbalancedDatasetUnderSampler(
    labels=Data.labels, indices=test_idx, times=config["test_times"]
)


# batch_size = train_idx.size
batch_size = tr_idx.size
train_loader = DataLoader(
    Data, sampler=train_sampler, batch_size=batch_size, drop_last=True
)

for batch, (data, target, cor) in enumerate(train_loader, 1):
    if type(data) == type([]):
        data[0] = data[0]
        data[1] = data[1]
    #                if batch >= 39000:
    #                    print("batch = ", batch, "data[0].size", data[0].size(), "data[1].size", data[1].size())
    else:
        data = data

dshape = data.shape
dataflat = data.reshape([dshape[0], dshape[1] * dshape[2] * dshape[3]])

dataflat.shape
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=dataflat.shape[1], num=5)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}
print(random_grid)

params = {
    "n_estimators": 200,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": None,
    "bootstrap": False,
}
clf = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1, **params)
# rf = clf = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1)
# clf = RandomizedSearchCV(
#    estimator=rf,
#    param_distributions=random_grid,
#    n_iter=10,
#    cv=3,
#    verbose=2,
#    random_state=42,
#    n_jobs=-1,
# )

clf.fit(dataflat, target)

# clf.best_params_
# {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}

clf.score(dataflat, target)


predtrain = clf.predict(dataflat)
cm = confusion_matrix(target, predtrain)
print(cm)
cr = classification_report(target, predtrain)
print(cr)

# batch_size = test_idx.size
batch_size = 10000
test_loader = DataLoader(
    Data, sampler=test_sampler, batch_size=batch_size, drop_last=True
)

for batch, (datatest, targettest, cortest) in enumerate(test_loader, 1):
    if type(data) == type([]):
        datatest[0] = datatest[0]
        datatest[1] = datatest[1]
    #                if batch >= 39000:
    #                    print("batch = ", batch, "data[0].size", data[0].size(), "data[1].size", data[1].size())
    else:
        datatest = datatest

dshape = datatest.shape
dataflattest = datatest.reshape([dshape[0], dshape[1] * dshape[2] * dshape[3]])

predtest = clf.predict(dataflattest)
cm = confusion_matrix(targettest, predtest)
print(cm)
cr = classification_report(targettest, predtest)
print(cr)


importances = clf.feature_importances_
std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

# import pandas as pd
# import matplotlib.pyplot as plt
#
# forest_importances = pd.Series(importances)
#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()


# Test 2019 data 2020 labels
start_year = 19
end_year = 19

Data = with_DSM(
    size=int(size / 2),
    start_year=start_year,
    end_year=end_year,
    wherepath=wherepath,
    DSM=DSM,
    type=hyperparameter_defaults["modeltype"],
)

test_idx = np.load(wherepath + "/" + "Test2D_idx%d.npy" % (end_year))
test_sampler = ImbalancedDatasetUnderSampler(
    labels=Data.labels, indices=test_idx, times=config["test_times"]
)
batch_size = 10000
test_loader = DataLoader(
    Data, sampler=test_sampler, batch_size=batch_size, drop_last=True
)

for batch, (datatest, targettest, cortest) in enumerate(test_loader, 1):
    if type(data) == type([]):
        datatest[0] = datatest[0]
        datatest[1] = datatest[1]
    #                if batch >= 39000:
    #                    print("batch = ", batch, "data[0].size", data[0].size(), "data[1].size", data[1].size())
    else:
        datatest = datatest


dshape = datatest.shape
dataflattest = datatest.reshape([dshape[0], dshape[1] * dshape[2] * dshape[3]])

predtest = clf.predict(dataflattest)
cm = confusion_matrix(targettest, predtest)
print(cm)
cr = classification_report(targettest, predtest)
print(cr)
