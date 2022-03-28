"""
SCRIPT FOR TRAINING 3DCNN MODELS
"""
import os
import torch
import numpy as np
import time
from ConvRNN import *
from Training import *
from Data_maker_loader import *
from random import randint, uniform, choice

random.seed(300)

region = "Madre"
start = time.time()
server = "/rds/general/user/jgb116/home/satellite/satellite/junin"
# server = '/rds/general/project/aandedemand/live/satellite/junin'

# WHERE TO IMPORT DATA FROM
# wherepath = server + '/data_reduced/tensors'
# savepath = server + '/data_reduced/tensors'
wherepath = server + "/data/" + region
savepath = server + "/data/" + region + "/out"
if not os.path.exists(savepath):
    os.makedirs(savepath)

# WHERE TO SAVE MODEL CHECKPOINT
modelpath = server + "/models/" + region + "_models/3D"
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

# WHERE TO SAVE IMAGES TRACKING TRAINING PROCESS
picspath = server + "/models/" + region + "_models/3D/pics"
if not os.path.exists(picspath):
    os.makedirs(picspath)

# WHERE TO SAVE MODEL PERFORMANCE OF EACH JOB FOR TRAIN, VAL AND TEST DATA
file = server + "/models/" + region + "_models/3D/grid_summary/ConvRNN.Conv_3D.mem.txt"
if not os.path.exists(os.path.dirname(file)):
    os.makedirs(os.path.dirname(file))

if __name__ == "__main__":
    # Set training time period
    start_year = 14
    end_year = 17

    # set CNN model parameters
    # size = 45
    sizes = [45, 49, 55, 59]
    DSM = False

    # CHOOSE THE INPUT DIMENSIONS - No DSM is (2,8). With DSM is (3,8)
    if DSM:
        input_dim = (3, 8)
    else:
        input_dim = (2, 8)

    hidden_dim = [64, 128, 128]
    kernel_size = [(3, 3), (2, 3, 3), (3, 3)]
    stride = [(2, 2), (1, 1), (1, 1), (1, 1)]
    padding = [0, 0, 0, 0]
    # dropout = 0.4
    dropouts = [0.2, 0.3, 0.4, 0.5]  # 3 options
    levels = [10]

    # set ratios of 0:1 labels in Train and Validation data sets
    train_times = 4
    test_times = 4

    # set criteria for Early stopping
    AUC = True
    BCE_Wloss = False
    FNcond = False
    # set parameters for the cost of the confusion matrix
    w = 10  # weights on the False Negative Rate
    perc = (100 * train_times) / (
        train_times + 1
    )  # the percentile to for treshold selection. Advisable to be 100*times/(times+1)

    # Weight parameter for the weighted BCE/CE loss
    pos_weight = 2
    # pos_weight = torch.tensor([1, pos_weight], dtype=torch.float, device="cuda:0")

    # Adam optimiser parameters:
    lrs = [0.000005, 0.00001, 0.00003, 0.00006, 0.00008, 0.0001, 0.0003, 0.001]
    weight_decay = 0

    # Early Stopping parameters
    n_splits = 5
    n_epochs = 20
    patience = 7
    training_time = (
        23.5  # Time in hours (needs to be less than 24 for GPUs in Imperial HPC)
    )

    # train_model parameters for debbuging and time regulations
    stop_batch = None
    print_batch = 1000
    batch_size = 32
    # batch_size = 1024

    # To explote different parameters in parallel
    if "PBS_ARRAY_INDEX" in os.environ:
        job = int(os.environ["PBS_ARRAY_INDEX"])
    else:
        job = 3
    if "PBS_JOBID" in os.environ:
        job_id = str(os.environ["PBS_JOBID"])
    else:
        job_id = "1"

    if job == 1:
        print("default params")
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 2:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 3:
        #        end_year = 17
        #        print("Settings to make it run a lot faster for debugging purposes...")
        #        input_dim=(3,8)
        #        hidden_dim=(16,32,32)
        #        kernel_size=((5,5),(2,5,5),(5,5))
        #        levels=(10,)
        #        training_time = 30
        #        stop_batch = 10
        #        print_batch = 1
        #        batch_size = 64
        #        stride = [(2,2),(1,1),(1,1),(1,1)]
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 4:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 5:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 6:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 7:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 8:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 9:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 10:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 11:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    if job == 12:
        lr = choice(lrs)
        size = (2 * randint(20, 40)) + 1
        dropout = round(uniform(0.1, 0.8), 1)

    # Set up model
    if (start_year - end_year) % 2 == 0:
        model = Conv_3Dodd(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            levels=levels,
            dropout=dropout,
        )
    else:
        model = Conv_3Deven(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            levels=levels,
            dropout=dropout,
        )
    model = torch.nn.DataParallel(model)

    # Set loss criterion and optimiser type
    # criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight = pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=torch.tensor(pos_weight)
    )
    optimiser = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Load data
    Data = with_DSM(
        size=int(size / 2),
        start_year=start_year,
        end_year=end_year,
        wherepath=wherepath,
        DSM=DSM,
    )

    if not (
        os.path.isfile(wherepath + "/" + "Train3D_idx%d.npy" % (end_year))
        & os.path.isfile(wherepath + "/" + "Test3D_idx%d.npy" % (end_year))
    ):
        print("Creating indexes split")
        train_idx, test_idx = train_test_split(
            np.arange(len(Data.labels)),
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=Data.labels,
        )
        np.save(wherepath + "/" + "Train3D_idx%d.npy" % (end_year), train_idx)
        np.save(wherepath + "/" + "Test3D_idx%d.npy" % (end_year), test_idx)
    else:
        print("loading: " + wherepath + "/" + "Train3D_idx%d.npy" % (end_year))
        train_idx = np.load(wherepath + "/" + "Train3D_idx%d.npy" % (end_year))
        test_idx = np.load(wherepath + "/" + "Test3D_idx%d.npy" % (end_year))

    # Set train and test samplers
    train_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=train_idx, times=train_times
    )
    test_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=test_idx, times=test_times
    )

    # Print model and training details
    print(
        "Model:",
        str(type(model))[8:-2],
        "\nPeriod 20%d-20%d -> 20%d" % (start_year, end_year, end_year + 1),
    )
    print(
        "\t% deforested pixels in train:",
        train_sampler.count[1] / sum(train_sampler.count),
    )
    print(
        "\t% deforested pixels in val:", test_sampler.count[1] / sum(test_sampler.count)
    )
    print("Job: ", job_id)
    print("DSM:", DSM)
    print("\nHyperparameters: ")
    print("\tImage size: %d" % (size))
    print("\tHidden dim: ", hidden_dim)
    print("\tDropout: ", dropout)
    print(
        "\tTrain and Val ratios of 0:1 labels: 1:%d ; 1:%d " % (train_times, test_times)
    )
    print(
        "\tADAM optimizer parameters: lr=%.7f, weight decay=%.2f, batch size=%d"
        % (lr, weight_decay, batch_size)
    )
    print("\tBCEWithLogitsLoss pos_weights = %.2f" % (pos_weight))
    print("\tn_epochs = %d with patience of %d epochs" % (n_epochs, patience))
    print("\tCross Validation with n_splits = %d " % (n_splits))
    print(
        "\tIf to use BCEWithLogitsLoss as an early stop criterion :",
        ((not AUC) & (not FNcond)),
    )
    print("\tIf to use AUC as an early stop criterion :", AUC)
    print("\tIf to use cost = FP+w*FN / TP+FP+w*FN+TN as an early stop criterion")
    print(
        "\twith w = %d and treshhold = the %d percentile of the output" % (w, perc),
        FNcond,
    )
    print("\nModel: \n", model)
    print("\nCriterion: \n", criterion)
    print("\nOptimiser: \n", optimiser)

    # Initiate training routine
    (
        model,
        train_loss,
        valid_loss,
        AUCs_train,
        AUCs_val,
        costs_train,
        costs_val,
        name,
    ) = train_model(
        Data=Data,
        model=model,
        sampler=train_sampler,
        criterion=criterion,
        optimiser=optimiser,
        patience=patience,
        n_epochs=n_epochs,
        n_splits=n_splits,
        batch_size=batch_size,
        stop_batch=stop_batch,
        print_batch=print_batch,
        training_time=training_time,
        w=w,
        FNcond=FNcond,
        AUC=AUC,
        job=job_id,
        path=modelpath,
    )
    # Produce graphs
    visualize(
        train=train_loss,
        valid=valid_loss,
        name="BCEloss",
        modelname=name,
        best="min",
        path=picspath,
    )
    visualize(
        train=AUCs_train,
        valid=AUCs_val,
        name="AUC",
        modelname=name,
        best="max",
        path=picspath,
    )
    visualize(
        train=costs_train,
        valid=costs_val,
        name="Cost",
        modelname=name,
        best="min",
        path=picspath,
    )

    test_loss, test_AUC, test_cost = test_model(
        model=model,
        Data=Data,
        criterion=criterion,
        w=w,
        perc=perc,
        test_sampler=test_sampler,
        batch_size=batch_size,
        stop_batch=stop_batch,
        name=name,
        path=picspath,
    )

    write_report(
        name=name,
        job_id=job_id,
        train_loss=train_loss,
        valid_loss=valid_loss,
        test_loss=test_loss,
        AUCs_train=AUCs_train,
        AUCs_val=AUCs_val,
        test_AUC=test_AUC,
        costs_train=costs_train,
        costs_val=costs_val,
        test_cost=test_cost,
        file=file,
        FNcond=FNcond,
        AUC=AUC,
    )

    print("\n\nEND!Total time (in h):", (time.time() - start) / 3600)
