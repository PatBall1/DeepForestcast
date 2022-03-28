"""
SCRIPT FOR TRAINING 2DCNN MODELS
- Wandb for logging runs / hyperparameter tuning
"""
import os
import torch
import numpy as np
import time
import wandb
from sklearn.model_selection import train_test_split
from CNN import CNNmodel
from Training import test_model, train_model
from Training import visualize, write_report, ImbalancedDatasetUnderSampler
from Data_maker_loader import with_DSM

# random.seed(300)
start = time.time()
server = "/rds/user/jgcb3/hpc-work/forecasting/junin/"

hyperparameter_defaults = dict(
    region="Madre",
    kernel_size=[(5, 5), (5, 5), (3, 3), (3, 3)],
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
    padding=[0, 0, 0, 0],
    size=36,
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
# Are GPU's available to use?
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# WandB - Initialize new run
wandb.init(config=hyperparameter_defaults, project="forecasting2D", entity="patball")
# Access all hyperparameter values through wandb.config
config = wandb.config

# Years
start_year = config["start_year"]
end_year = config["end_year"]

# WHERE TO IMPORT DATA FROM
wherepath = server + "/data/" + config["region"]
savepath = server + "/data/" + config["region"] + "/out"
if not os.path.exists(savepath):
    os.makedirs(savepath)

# WHERE TO SAVE MODEL CHECKPOINT
modelpath = server + "/models/" + config["region"] + "_models/2D"
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

# WHERE TO SAVE IMAGES TRACKING TRAINING PROCESS
picspath = server + "/models/" + config["region"] + "_models/2D/pics"
if not os.path.exists(picspath):
    os.makedirs(picspath)

# WHERE TO SAVE MODEL PERFORMANCE OF EACH JOB FOR TRAIN, VAL AND TEST DATA
file = server + "/models/" + config["region"] + "_models/2D/grid_summary/2DCNN.txt"
if not os.path.exists(os.path.dirname(file)):
    os.makedirs(os.path.dirname(file))


if __name__ == "__main__":
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        job = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        job = 3
    if "SLURM_JOBID" in os.environ:
        job_id = str(os.environ["SLURM_JOBID"])
    else:
        job_id = "1"

    # the percentile to for treshold selection. Advisable to be 100*times/(times+1)
    perc = (100 * config["train_times"]) / (config["train_times"] + 1)
    # set CNN model parameters
    # size = 45
    DSM = False

    # CHOOSE THE INPUT DIMENSIONS - No DSM is (2,8). With DSM is (3,8)
    if DSM:
        input_dim = 11
    else:
        input_dim = 10

    # train_model parameters for debbuging and time regulations

    # To exploite different parameters in parallel
    # Set up model

    hidden_dim = [
        config["hidden_dim1"],
        config["hidden_dim2"],
        config["hidden_dim3"],
        config["hidden_dim4"],
    ]
    model = CNNmodel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=config["kernel_size"],
        levels=[config["levels"]],
        dropout=config["dropout"],
        # start_year=config["start_year"],
        # end_year=config["end_year"],
        stride=config["stride"],
        padding=config["padding"],
    )
    model = torch.nn.DataParallel(model)

    # Set loss criterion and optimiser type
    # criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight = pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=torch.tensor(config["pos_weight"])
    )
    optimiser = torch.optim.Adam(
        params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    # Necessary if you want to restart training with pre-trained model
    if "CHECKPOINT" in os.environ:
        checkpoint = os.environ["CHECKPOINT"]
        print("Loading checkpoint..." + checkpoint)
        modelname = checkpoint.split("/", -1)[-1]
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        print("Checkpoint loaded")
    # Load data
    Data = with_DSM(
        size=int(config["size"] / 2),
        start_year=config["start_year"],
        end_year=config["end_year"],
        wherepath=wherepath,
        DSM=DSM,
        type=config["modeltype"],
    )
    # if not (
    #    os.path.isfile(wherepath + "/" + "Train_idx%d.npy" % (end_year))
    #    & os.path.isfile(wherepath + "/" + "Test_idx%d.npy" % (end_year))
    # ):
    #    print("Creating indexes split")
    #    train_idx, test_idx = train_test_split(
    #        np.arange(len(Data.labels)),
    #        test_size=0.2,
    #        random_state=42,
    #        shuffle=True,
    #        stratify=Data.labels,
    #    )
    #    np.save(wherepath + "/" + "Train_idx%d.npy" % (end_year), train_idx)
    #    np.save(wherepath + "/" + "Test_idx%d.npy" % (end_year), test_idx)
    # else:
    #    print("loading: " + wherepath + "/" + "Train_idx%d.npy" % (end_year))
    #    train_idx = np.load(wherepath + "/" + "Train_idx%d.npy" % (end_year))
    #    test_idx = np.load(wherepath + "/" + "Test_idx%d.npy" % (end_year))
    #    test_sampler = ImbalancedDatasetUnderSampler(
    #    labels=Data.labels, indices=test_idx, times=config["test_times"]
    # )
    if not (
        os.path.isfile(wherepath + "/" + "Train2D_idx%d.npy" % (config["end_year"]))
        & os.path.isfile(wherepath + "/" + "Test2D_idx%d.npy" % (config["end_year"]))
    ):
        print("Creating indexes split")
        train_idx, test_idx = train_test_split(
            np.arange(len(Data.labels)),
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=Data.labels,
        )
        np.save(wherepath + "/" + "Train2D_idx%d.npy" % (config["end_year"]), train_idx)
        np.save(wherepath + "/" + "Test2D_idx%d.npy" % (config["end_year"]), test_idx)
    else:
        print(
            "loading: " + wherepath + "/" + "Train2D_idx%d.npy" % (config["end_year"])
        )
        train_idx = np.load(
            wherepath + "/" + "Train2D_idx%d.npy" % (config["end_year"])
        )
        test_idx = np.load(wherepath + "/" + "Test2D_idx%d.npy" % (config["end_year"]))

    # Set train and test samplers
    train_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=train_idx, times=config["train_times"]
    )
    test_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=test_idx, times=config["test_times"]
    )

    # Print model and training details
    print(
        "Model:",
        str(type(model))[8:-2],
        "\nPeriod 20%d-20%d -> 20%d"
        % (config["start_year"], config["end_year"], config["end_year"] + 1),
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
    print("\tImage size: %d" % (config["size"]))
    print("\tHidden dim: ", hidden_dim)
    print("\tDropout: ", config["dropout"])
    print(
        "\tTrain and Val ratios of 0:1 labels: 1:%d ; 1:%d "
        % (config["train_times"], config["test_times"])
    )
    print(
        "\tADAM optimizer parameters: lr=%.7f, weight decay=%.2f, batch size=%d"
        % (config["lr"], config["weight_decay"], config["batch_size"])
    )
    print("\tBCEWithLogitsLoss pos_weights = %.2f" % (config["pos_weight"]))
    print(
        "\tn_epochs = %d with patience of %d epochs"
        % (config["n_epochs"], config["patience"])
    )
    print("\tCross Validation with n_splits = %d " % (config["n_splits"]))
    print(
        "\tIf to use BCEWithLogitsLoss as an early stop criterion :",
        ((not config["AUC"]) & (not config["FNcond"])),
    )
    print("\tIf to use AUC as an early stop criterion :", config["AUC"])
    print("\tIf to use cost = FP+w*FN / TP+FP+w*FN+TN as an early stop criterion")
    print(
        "\twith w = %d and treshhold = the %d percentile of the output"
        % (config["w"], perc),
        config["FNcond"],
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
        patience=config["patience"],
        n_epochs=config["n_epochs"],
        n_splits=config["n_splits"],
        batch_size=config["batch_size"],
        stop_batch=config["stop_batch"],
        print_batch=config["print_batch"],
        training_time=config["training_time"],
        w=config["w"],
        FNcond=config["FNcond"],
        AUC=config["AUC"],
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
        w=config["w"],
        perc=perc,
        test_sampler=test_sampler,
        batch_size=config["batch_size"],
        stop_batch=config["stop_batch"],
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
        FNcond=config["FNcond"],
        AUC=config["AUC"],
    )

    print("\n\nEND!Total time (in h):", (time.time() - start) / 3600)
