import torch
import numpy as np
from CNN import *
from Training import *
from Testing import *
from Data_maker_loader import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
import time

server = "/rdsgpfs/general/user/kpp15/home/Hansen"
# from where to load tensors for data
wherepath = server + "/data/raster/tensors"
# Where to get and save tif map
sourcepath = server + "/data/raster/MadreTiff"
# Where to get model checkpoint
modelpath = server + "/deforestation_forecasting/models"
checkpoint = modelpath + "/CNN.CNNmodel/CNN.CNNmodel_3.7.19_23.47_315834[9].pbs.pt"
modelname = checkpoint.split("/", -1)[-1]
# Where to save Test_Roc
picspath = server + "/deforestation_forecasting/models/pics"
file = server + "/deforestation_forecasting/models/grid_summary/CNN.CNNmodel.txt"

if __name__ == "__main__":

    start = time.time()

    # Set all parameters

    # Set training time period
    start_year = 17
    end_year = 17

    # set CNN model parameeters
    size = 45
    DSM = True
    input_dim = 11

    hidden_dim = [128, 64, 64, 32]
    kernel_size = [(5, 5), (5, 5), (3, 3), (3, 3)]
    stride = [(2, 2), (1, 1), (1, 1), (1, 1)]
    padding = [0, 0, 0, 0]
    dropout = 0.2
    levels = [13]

    # set ratios of 0:1 labels in Train and Validation data sets
    train_times = 4
    test_times = 4

    # set criteria for Early stopping
    AUC = False
    BCE_Wloss = False
    FNcond = True
    # set parameters for the cost of the confussion matrix
    w = 10  # weights on the False Negative Rate
    perc = (100 * train_times) / (
        train_times + 1
    )  # the percentile to for treshhold selection. Advisable to be 100*times/(times+1)

    # Weight parameter for the weighted BCE loss
    pos_weight = 3

    # Adam optimiser parameters:
    lr = 0.0001
    weight_decay = 0

    # Early Stopping parameters
    n_splits = 5
    n_epochs = 10
    patience = 3

    # train_model parameters for debbuging and time regulations
    training_time = 60
    stop_batch = None
    print_batch = 200
    batch_size = 32

    model = CNNmodel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dropout=dropout,
        levels=levels,
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=torch.tensor(pos_weight)
    )
    optimiser = torch.optim.Adam(
        params=model.parameters(), lr=0.0001, weight_decay=weight_decay
    )

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    Data = with_DSM(
        size=int(size / 2),
        start_year=start_year,
        end_year=end_year,
        wherepath=wherepath,
        DSM=DSM,
    )

    if not (
        os.path.isfile(wherepath + "/" + "Train_idx%d.npy" % (end_year))
        & os.path.isfile(wherepath + "/" + "Test_idx%d.npy" % (end_year))
    ):
        print("Creating indexes split")
        train_idx, test_idx = train_test_split(
            np.arange(len(Data.labels)),
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=Data.labels,
        )
        np.save(wherepath + "Train_idx%d.npy" % (end_year), train_idx)
        np.save(wherepath + "Test_idx%d.npy" % (end_year), test_idx)
    else:
        train_idx = np.load(wherepath + "/" + "Train_idx%d.npy" % (end_year))
        test_idx = np.load(wherepath + "/" + "Test_idx%d.npy" % (end_year))

    train_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=train_idx, times=train_times
    )
    test_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=test_idx, times=test_times
    )

    job_id = modelname + f".transfer_learning_20{end_year+1:d}"
    # Print Summary of the training parameters
    # =========================================================================
    print(
        "Model:",
        modelname,
        "\nPeriod 20%d-20%d -> 20%d" % (start_year, end_year, end_year + 1),
    )
    print("New model:", job_id)
    print(
        "\t% deforested pixels in train:",
        train_sampler.count[1] / sum(train_sampler.count),
    )
    print(
        "\t% deforested pixels in val:", test_sampler.count[1] / sum(test_sampler.count)
    )
    print("\nHyperparameters: ")
    print("\tImage size: %d" % (size))
    print("\tHidden dim: ", hidden_dim)
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
        perc=perc,
        FNcond=FNcond,
        AUC=AUC,
        job=job_id,
        path=modelpath,
    )

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
