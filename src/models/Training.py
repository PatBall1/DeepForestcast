import numpy as np
import random
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sklearn as skl
import scikitplot
from scipy.special import expit as Sigmoid
import os
import time
import datetime
import matplotlib.pyplot as plt
import wandb


def train_model(
    Data,
    model,
    sampler,
    criterion,
    optimiser,
    patience=3,
    n_epochs=5,
    n_splits=5,
    batch_size=128,
    stop_batch=None,
    print_batch=32,
    training_time=12,
    w=10,
    perc=80,
    FNcond=True,
    AUC=False,
    job=1,
    path="~/junin/deforestation_forecasting/models",
):

    print("\nTRAINING ROUTINE COMMENCING")
    #############################################
    # Set up the variables to track the training #
    #############################################
    t_start = time.time()

    # Track device used
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    if use_cuda:
        print("Device accessed:", torch.cuda.get_device_name(0))
        print("Number of devices:", torch.cuda.device_count())

    require_sigmoid = isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss)
    # require_sigmoid = isinstance(criterion,torch.nn.modules.loss.CrossEntropyLoss)
    # to track the training losses as the model trains and validate on each epoch
    train_losses = []
    valid_losses = []

    # to track the average losses as the model trains and validate per epoch
    avg_train_losses = []
    avg_valid_losses = []

    # to track the probabilities the model give on the train and validate data on each epoch
    train_outputs = []
    val_outputs = []

    # to track the true labels of the inputs when model trains and validate on each epoch
    train_targets = []
    val_targets = []

    # to track the cost as the model trains and validate on each epoch
    costs_train = []
    costs_val = []

    # to track the AUC as the model trains and validate on each epoch
    AUCs_train = []
    AUCs_val = []

    # initialize unique model name
    d = datetime.datetime.now()
    print("\nDate:", d)
    model_type = str(type(model))[8:-2]
    name = (
        str(type(model))[8:-2]
        + f"_{d.day:d}"
        + f".{d.month:d}"
        + f".{str(d.year)[-2:]}"
        + f"_{d.hour:d}"
        + f".{d.minute:d}"
        + "_"
        + str(job)
        + ".pt"
    )

    if not os.path.exists(path + "/" + model_type):
        os.mkdir(path + "/" + model_type)
        print("Directory ", path + "/" + model_type, " Created ")

    path = os.path.join(path, model_type, name)

    print("\nCheckpoint saved at:", path)

    early_stopping = EarlyStopping(patience=patience, path=path)
    skf = StratifiedKFold(n_splits=n_splits)
    ls = list(
        skf.split(
            sampler.classIndexes_unsampled, Data.labels[sampler.classIndexes_unsampled]
        )
    )
    print("\nStart training...")
    ###############################
    # Start tracking the training #
    ###############################

    for epoch in range(0, n_epochs):
        fold = epoch % n_splits
        epoch = epoch + 1
        tr, val = ls[fold]
        tr_idx = sampler.classIndexes_unsampled[tr]
        val_idx = sampler.classIndexes_unsampled[val]
        print(
            "\n\nEpoch %d, Fold %d Train size %d; Val size %d"
            % (epoch, fold + 1, len(tr_idx), len(val_idx))
        )

        train_Sampler = SubsetRandomSampler(tr_idx)
        val_Sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(
            Data, sampler=train_Sampler, batch_size=batch_size, drop_last=True
        )
        val_loader = DataLoader(
            Data, sampler=val_Sampler, batch_size=batch_size, drop_last=True
        )

        ###################
        # train the model #
        ###################
        model.train()
        train_start = time.time()
        print("\n\tLearning:...")
        for batch, (data, target, cor) in enumerate(train_loader, 1):

            if type(data) == type([]):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
            #                if batch >= 39000:
            #                    print("batch = ", batch, "data[0].size", data[0].size(), "data[1].size", data[1].size())
            else:
                data = data.to(device)
            #                if batch >= 39000:
            #                    print("batch = ", batch, "data.size", data.size())
            target = target.to(device)
            cor = cor.to(device)

            optimiser.zero_grad()
            output = model.forward(data, sigmoid=not require_sigmoid)
            loss = criterion(output, target.float())
            loss.backward()
            optimiser.step()
            # record training loss, predictions and targets
            train_losses.append(loss.cpu().item())
            train_outputs.append(list(output.cpu().data))
            train_targets.append(list(target.cpu().data))
            # =============================================================================
            if stop_batch:
                if batch == stop_batch:
                    break
            # =============================================================================
            if print_batch:
                if batch % print_batch == 0:
                    print(
                        "\tEpoch:",
                        epoch,
                        "\tBatch:",
                        batch,
                        "\tTraining Loss:",
                        loss.item(),
                    )
                    wandb.log({"bat_train_loss": loss.item()})
            # =============================================================================
        train_outputs = sum(train_outputs, [])
        train_targets = sum(train_targets, [])
        print("\tStop learning")
        t_time = time.time() - train_start
        print(
            "\n\tTime to load %d train batches of size %d : %3.4f hours "
            % (batch, batch_size, t_time / (3600))
        )

        #####################################
        # Report AUC and the CM on Train data#
        #####################################
        print("\n\tAUC and CM on Train data")
        auc_t, cost_train = AUC_CM(
            train_targets, train_outputs, w, perc, sigmoid=require_sigmoid
        )

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        val_start = time.time()
        for batch, (data, target, cor) in enumerate(val_loader, 1):

            if type(data) == type([]):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
            else:
                data = data.to(device)
            target = target.to(device)
            cor = cor.to(device)

            output = model.forward(data, sigmoid=not require_sigmoid)
            loss = criterion(output, target.float())
            wandb.log({"bat_val_loss": loss.cpu().item()})
            valid_losses.append(loss.cpu().item())
            val_outputs.append(list(output.cpu().data))
            val_targets.append(list(target.cpu().data))
            # =============================================================================
            if stop_batch:
                if batch == stop_batch:
                    break
            # =============================================================================
        val_outputs = sum(val_outputs, [])
        val_targets = sum(val_targets, [])
        v_time = time.time() - val_start
        print(
            "\n\tTime to load %d val batches of size %d : %3.4f hours "
            % (batch, batch_size, v_time / (3600))
        )

        ###############################
        # Report AUC and CM on Val data#
        ###############################
        print("\tAUC and CM on Val data")
        auc_v, cost_val = AUC_CM(
            val_targets, val_outputs, w, perc, sigmoid=require_sigmoid
        )

        # Save results of the epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        costs_train.append(cost_train)
        costs_val.append(cost_val)
        AUCs_train.append(auc_t)
        AUCs_val.append(auc_v)
        wandb.log(
            {
                "train_loss": train_loss,
                "validation_loss": valid_loss,
                "train_cost": cost_train,
                "validation_cost": cost_val,
                "train_AUC": auc_t,
                "validation_AUC": auc_v,
                "epoch": epoch,
            }
        )
        # clear lists to track next epoch
        train_losses = []
        train_targets = []
        train_outputs = []

        valid_losses = []
        val_outputs = []
        val_targets = []

        # print epoch summary
        epoch_len = len(str(n_epochs))
        time_mid = time.time()
        total_time = time_mid - t_start
        total_time = total_time / 3600  # Total time in hours rather than seconds
        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"valid_loss: {valid_loss:.5f}"
            + "\n"
            + f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f" train AUC: {auc_t:3.5f}"
            + f" val AUC: {auc_v:3.5f}"
            + "\n"
            + f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f" train cost: {cost_train:.5f}"
            + f" val cost: {cost_val:.5f}"
            + "\n"
            + f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f" hours needed: {total_time: 4.4f}"
        )
        print("\n\nEpoch summary: ")
        print(print_msg)

        # Time elapsed
        t_elapsed = time.time() - t_start
        print("\n\tTime elapsed: %3.4f hours " % (t_elapsed / (3600)))

        # Check early stopping criteria
        # early_stopping needs the validation loss/Weighted error loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if AUC:
            early_stopping(-auc_v, model, optimiser)
        elif FNcond:
            early_stopping(cost_val, model, optimiser)
        else:
            early_stopping(valid_loss, model, optimiser)

        if early_stopping.early_stop:
            print("\nEarly stopping!")
            break

        # Check total training time to here
        if total_time > training_time:
            print("\nTime exceeded!")
            break

        # Check if all folds are passed
        if (fold == n_splits - 1) & (epoch < n_epochs):
            print("Folds exceeded. Adding new 0 labels...")
            sampler.update()
            ls = list(
                skf.split(
                    sampler.classIndexes_unsampled,
                    Data.labels[sampler.classIndexes_unsampled],
                )
            )
            print("New 0 labels added!")

    # load the last checkpoint with the best model
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return (
        model,
        avg_train_losses,
        avg_valid_losses,
        AUCs_train,
        AUCs_val,
        costs_train,
        costs_val,
        name[:-3],
    )


"""
def train_model(
    Data,
    model,
    sampler,
    criterion,
    optimiser,
    patience=3,
    n_epochs=5,
    n_splits=5,
    batch_size=128,
    stop_batch=None,
    print_batch=32,
    training_time=12,
    w=10,
    perc=80,
    FNcond=True,
    AUC=False,
    job=1,
    path="~/junin/deforestation_forecasting/models",
):

    print("\nTRAINING ROUTINE COMMENCING")
    #############################################
    # Set up the variables to track the training #
    #############################################
    t_start = time.time()

    # Track device used
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    if use_cuda:
        print("Device accessed:", torch.cuda.get_device_name(0))
        print("Number of devices:", torch.cuda.device_count())

    require_sigmoid = isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss)
    # require_sigmoid = isinstance(criterion,torch.nn.modules.loss.CrossEntropyLoss)
    # to track the training losses as the model trains and validate on each epoch
    train_losses = []
    valid_losses = []

    # to track the average losses as the model trains and validate per epoch
    avg_train_losses = []
    avg_valid_losses = []

    # to track the probabilities the model give on the train and validate data on each epoch
    train_outputs = []
    val_outputs = []

    # to track the true labels of the inputs when model trains and validate on each epoch
    train_targets = []
    val_targets = []

    # to track the cost as the model trains and validate on each epoch
    costs_train = []
    costs_val = []

    # to track the AUC as the model trains and validate on each epoch
    AUCs_train = []
    AUCs_val = []

    # initialize unique model name
    d = datetime.datetime.now()
    print("\nDate:", d)
    model_type = str(type(model))[8:-2]
    name = (
        str(type(model))[8:-2]
        + f"_{d.day:d}"
        + f".{d.month:d}"
        + f".{str(d.year)[-2:]}"
        + f"_{d.hour:d}"
        + f".{d.minute:d}"
        + "_"
        + str(job)
        + ".pt"
    )

    if not os.path.exists(path + "/" + model_type):
        os.mkdir(path + "/" + model_type)
        print("Directory ", path + "/" + model_type, " Created ")

    path = os.path.join(path, model_type, name)

    print("\nCheckpoint saved at:", path)

    early_stopping = EarlyStopping(patience=patience, path=path)
    skf = StratifiedKFold(n_splits=n_splits)
    ls = list(
        skf.split(
            sampler.classIndexes_unsampled, Data.labels[sampler.classIndexes_unsampled]
        )
    )
    print("\nStart training...")
    ###############################
    # Start tracking the training #
    ###############################

    for epoch in range(0, n_epochs):
        fold = epoch % n_splits
        epoch = epoch + 1
        tr, val = ls[fold]
        tr_idx = sampler.classIndexes_unsampled[tr]
        val_idx = sampler.classIndexes_unsampled[val]
        print(
            "\n\nEpoch %d, Fold %d Train size %d; Val size %d"
            % (epoch, fold + 1, len(tr_idx), len(val_idx))
        )

        train_Sampler = SubsetRandomSampler(tr_idx)
        val_Sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(
            Data, sampler=train_Sampler, batch_size=batch_size, drop_last=True
        )
        val_loader = DataLoader(
            Data, sampler=val_Sampler, batch_size=batch_size, drop_last=True
        )

        ###################
        # train the model #
        ###################
        model.train()
        train_start = time.time()
        print("\n\tLearning:...")
        for batch, (data, target, cor) in enumerate(train_loader, 1):

            if type(data) == type([]):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
            #                if batch >= 39000:
            #                    print("batch = ", batch, "data[0].size", data[0].size(), "data[1].size", data[1].size())
            else:
                data = data.to(device)
            #                if batch >= 39000:
            #                    print("batch = ", batch, "data.size", data.size())
            target = target.to(device)
            cor = cor.to(device)

            optimiser.zero_grad()
            output = model.forward(data, sigmoid=not require_sigmoid)
            loss = criterion(output, target.float())
            loss.backward()
            optimiser.step()
            # record training loss, predictions and targets
            train_losses.append(loss.cpu().item())
            train_outputs.append(list(output.cpu().data))
            train_targets.append(list(target.cpu().data))
            # =============================================================================
            if stop_batch:
                if batch == stop_batch:
                    break
            # =============================================================================
            if print_batch:
                if batch % print_batch == 0:
                    print(
                        "\tEpoch:",
                        epoch,
                        "\tBatch:",
                        batch,
                        "\tTraining Loss:",
                        loss.item(),
                    )
            # =============================================================================
        train_outputs = sum(train_outputs, [])
        train_targets = sum(train_targets, [])
        print("\tStop learning")
        t_time = time.time() - train_start
        print(
            "\n\tTime to load %d train batches of size %d : %3.4f hours "
            % (batch, batch_size, t_time / (3600))
        )

        #####################################
        # Report AUC and the CM on Train data#
        #####################################
        print("\n\tAUC and CM on Train data")
        auc_t, cost_train = AUC_CM(
            train_targets, train_outputs, w, perc, sigmoid=require_sigmoid
        )

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        val_start = time.time()
        for batch, (data, target, cor) in enumerate(val_loader, 1):

            if type(data) == type([]):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
            else:
                data = data.to(device)
            target = target.to(device)
            cor = cor.to(device)

            output = model.forward(data, sigmoid=not require_sigmoid)
            loss = criterion(output, target.float())
            valid_losses.append(loss.cpu().item())
            val_outputs.append(list(output.cpu().data))
            val_targets.append(list(target.cpu().data))
            # =============================================================================
            if stop_batch:
                if batch == stop_batch:
                    break
            # =============================================================================
        val_outputs = sum(val_outputs, [])
        val_targets = sum(val_targets, [])
        v_time = time.time() - val_start
        print(
            "\n\tTime to load %d val batches of size %d : %3.4f hours "
            % (batch, batch_size, v_time / (3600))
        )

        ###############################
        # Report AUC and CM on Val data#
        ###############################
        print("\tAUC and CM on Val data")
        auc_v, cost_val = AUC_CM(
            val_targets, val_outputs, w, perc, sigmoid=require_sigmoid
        )

        # Save results of the epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        costs_train.append(cost_train)
        costs_val.append(cost_val)
        AUCs_train.append(auc_t)
        AUCs_val.append(auc_v)
        wandb.log(
            {
                "Train loss": train_loss,
                "Validation loss": cost_train,
                "Train cost": cost_train,
                "Validation cost": valid_loss,
                "Train AUC": auc_t,
                "Validation AUC": auc_v,
                "Epoch": epoch,
            }
        )
        # clear lists to track next epoch
        train_losses = []
        train_targets = []
        train_outputs = []

        valid_losses = []
        val_outputs = []
        val_targets = []

        # print epoch summary
        epoch_len = len(str(n_epochs))
        time_mid = time.time()
        total_time = time_mid - t_start
        total_time = total_time / 3600  # Total time in hours rather than seconds
        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"valid_loss: {valid_loss:.5f}"
            + "\n"
            + f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f" train AUC: {auc_t:3.5f}"
            + f" val AUC: {auc_v:3.5f}"
            + "\n"
            + f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f" train cost: {cost_train:.5f}"
            + f" val cost: {cost_val:.5f}"
            + "\n"
            + f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f" hours needed: {total_time: 4.4f}"
        )
        print("\n\nEpoch summary: ")
        print(print_msg)

        # Time elapsed
        t_elapsed = time.time() - t_start
        print("\n\tTime elapsed: %3.4f hours " % (t_elapsed / (3600)))

        # Check early stopping criteria
        # early_stopping needs the validation loss/Weighted error loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if AUC:
            early_stopping(-auc_v, model, optimiser)
        elif FNcond:
            early_stopping(cost_val, model, optimiser)
        else:
            early_stopping(valid_loss, model, optimiser)

        if early_stopping.early_stop:
            print("\nEarly stopping!")
            break

        # Check total training time to here
        if total_time > training_time:
            print("\nTime exceeded!")
            break

        # Check if all folds are passed
        if (fold == n_splits - 1) & (epoch < n_epochs):
            print("Folds exceeded. Adding new 0 labels...")
            sampler.update()
            ls = list(
                skf.split(
                    sampler.classIndexes_unsampled,
                    Data.labels[sampler.classIndexes_unsampled],
                )
            )
            print("New 0 labels added!")

    # load the last checkpoint with the best model
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return (
        model,
        avg_train_losses,
        avg_valid_losses,
        AUCs_train,
        AUCs_val,
        costs_train,
        costs_val,
        name[:-3],
    )
"""


def test_model(
    model, Data, criterion, w, perc, test_sampler, batch_size, stop_batch, name, path
):
    print("\nTESTING ROUTINE COMMENCING")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()  # prep model for evaluation

    Test_sampler = SubsetRandomSampler(test_sampler.classIndexes_unsampled)
    test_loader = DataLoader(
        Data, sampler=Test_sampler, batch_size=batch_size, drop_last=True
    )
    require_sigmoid = isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss)
    # initialize lists to monitor test loss and accuracy
    test_losses = []
    output_total = []
    target_total = []

    test_start = time.time()
    for batch, (data, target, cor) in enumerate(test_loader):

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        target = target.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data, sigmoid=not require_sigmoid)
        # calculate the loss
        loss = criterion(output, target.float())
        test_losses.append(loss.item())
        output_total.append(list(output.cpu().data))
        target_total.append(list(target.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
    output_total = sum(output_total, [])
    target_total = sum(target_total, [])
    t_time = time.time() - test_start
    print(
        "\n\tTime to load %d test batches of size %d : %3.4f hours\n"
        % (batch, batch_size, t_time / (3600))
    )

    test_losses = np.average(test_losses)

    print("\tTest Loss: ", test_losses)
    test_AUC, test_cost = AUC_CM(
        target_total, output_total, w, perc, sigmoid=require_sigmoid
    )
    if require_sigmoid:
        output_total = np.array(Sigmoid(output_total))
    probas_per_class = np.stack((1 - output_total, output_total), axis=1)
    roc = scikitplot.metrics.plot_roc(np.array(target_total), probas_per_class)

    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " Created ")

    path = os.path.join(path, name + "_ROC.png")

    roc.get_figure().savefig(path, bbox_inches="tight")

    return test_losses, test_AUC, test_cost


class ImbalancedDatasetUnderSampler:
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices : a list of indices
        labels : The labels of the full data set
        times : the ratio of 1:0 labels
    """

    # self.classIndexes is a list of arrays coresponding to the indices of the input indeces for each class datapoints
    # That is, if:
    # indices = [3,6,7,8,11,15,45]
    # lables[indices] = [0,0,0,1,0,1,1]
    # self.classIndexes = [[0,1,2,4],[3,5,6]]
    #        indices[self.classIndexes[0]] =  [3,6,7,11]
    #            and labels[indices[self.classIndexes[0]]] = labels[3,6,7,11] = [0,0,0,0]
    #        indices[self.classIndexes[1]] =  [8,15,45]
    #            and labels[indices[self.classIndexes[1]] = labels[8,15,45] = [1,1,1]
    # self.count = [4,3]
    # self.min_class_len = 3

    # Finally,
    # self.classIndexes_unsampled is the undersampled subset of indices so that
    # labels[self.classIndexes_unsampled] has mean times/(times+1)

    def __init__(self, labels, indices, times):
        self.indices = indices
        self.labels = labels
        self.times = times
        self.classes = [0, 1]
        self.classIndexes = [
            (self.labels[self.indices] == cl).nonzero() for cl in self.classes
        ]
        self.count = [len(self.classIndexes[i]) for i in range(len(self.classes))]
        self.min_class_len = min(self.count)

        self.classIndexes_unsampled = []
        for i in range(len(self.classes)):
            if self.count[i] == self.min_class_len:
                self.classIndexes_unsampled.append(
                    list(self.classIndexes[i].numpy().squeeze())
                )
            else:
                indexes = random.sample(
                    set(self.classIndexes[i].numpy().squeeze()),
                    int(self.times * self.min_class_len),
                )
                self.classIndexes_unsampled.append(indexes)
        self.classIndexes_unsampled = sum(self.classIndexes_unsampled, [])
        self.classIndexes_unsampled = np.random.permutation(self.classIndexes_unsampled)
        self.classIndexes_unsampled = self.indices[self.classIndexes_unsampled]

    def update(self, times=None):

        if times:
            self.times = times

        self.classIndexes_unsampled = []
        for i in range(len(self.classes)):
            if self.count[i] == self.min_class_len:
                self.classIndexes_unsampled.append(
                    list(self.classIndexes[i].numpy().squeeze())
                )
            else:
                indexes = random.sample(
                    set(self.classIndexes[i].numpy().squeeze()),
                    int(self.times * self.min_class_len),
                )
                self.classIndexes_unsampled.append(indexes)
        self.classIndexes_unsampled = sum(self.classIndexes_unsampled, [])
        self.classIndexes_unsampled = np.random.permutation(self.classIndexes_unsampled)
        self.classIndexes_unsampled = self.indices[self.classIndexes_unsampled]

    def train_val_split(self, val_size=0.2):
        train_idx, val_idx = train_test_split(
            self.classIndexes_unsampled,
            test_size=val_size,
            shuffle=True,
            stratify=self.labels[self.classIndexes_unsampled],
        )
        return (train_idx, val_idx)

    def __iter__(self):
        return (i for i in self.classIndexes_unsampled)

    def __len__(self):
        return len(self.classIndexes_unsampled)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, path):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimiser):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimiser)

        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimiser)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimiser):
        """Saves model when validation loss decrease."""
        print(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "loss": val_loss,
            },
            self.path,
        )
        self.val_loss_min = val_loss


def AUC_CM(targets, outputs, w, perc, sigmoid):
    ##############################################
    # Reporting the AUC ROC and Confusion Matrix #
    ##############################################
    if sigmoid:
        outputs = Sigmoid(outputs)

    auc = skl.metrics.roc_auc_score(targets, outputs)
    tr = np.percentile(outputs, perc)
    matrix = skl.metrics.confusion_matrix(targets, (outputs > tr) * 1)
    cost = (matrix[0, 1] + w * matrix[1, 0]) / (np.sum(matrix) + (w - 1) * matrix[1, 0])
    print("\tAUC :", auc)
    print("\tPred:  ", "\t", 0, "\t\t", 1)
    print("\tTrue: 0", "\t", matrix[0, 0], "\t\t", matrix[0, 1])
    print("\tTrue: 1", "\t", matrix[1, 0], "\t\t", matrix[1, 1])
    print("\tAccuracy ", np.trace(matrix) / np.sum(matrix))
    print("\tTrue Positive Rate ", matrix[1, 1] / sum(matrix[1, :]))
    print("\tCost = FP+w*FN/TP+FP+w*FN+TN = %.4f (w = %d)" % (cost, w))
    print("\tTreshold %.4f when percentage of 0 predicted labels is %d" % (tr, perc))

    return auc, cost


def visualize(
    train,
    valid,
    name,
    modelname,
    best="min",
    path="/rdsgpfs/general/user/kpp15/home/Hansen/deforestation_forecasting/models/pics",
):
    print("\nReporting", name, "when training")
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train) + 1), train, label="Training " + name)
    plt.plot(range(1, len(valid) + 1), valid, label="Validation " + name)

    # find position of lowest validation loss
    if best == "min":
        bestposs = valid.index(min(valid)) + 1
    else:
        bestposs = valid.index(max(valid)) + 1

    plt.axvline(bestposs, linestyle="--", color="r", label="Early Stopping Checkpoint")
    plt.title(modelname + "_" + name + ".png")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim(0, len(train) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " Created ")
    path = os.path.join(path, modelname + "_" + name + ".png")
    fig.savefig(path, bbox_inches="tight")
    print("Figure saved at:", path)


def write_report(
    name,
    job_id,
    train_loss,
    valid_loss,
    test_loss,
    AUCs_train,
    AUCs_val,
    test_AUC,
    costs_train,
    costs_val,
    test_cost,
    file,
    FNcond=True,
    AUC=False,
):
    print("\nWRITING REPORT")

    if not os.path.exists(os.path.dirname(file)):
        os.mkdir(os.path.dirname(file))
        print("Directory ", os.path.dirname(file), " Created ")

    if FNcond:
        best_epoch = costs_val.index(min(costs_val))
    elif AUC:
        best_epoch = AUCs_val.index(max(AUCs_val))
    else:
        best_epoch = valid_loss.index(min(valid_loss))

    summary = (
        "| Job:" + job_id + " | Model:" + name + f" | Best epoch:{best_epoch+1:d} |"
    )
    Train = f"| Train loss:{train_loss[best_epoch]:.3f} | Train AUC:{AUCs_train[best_epoch]:.3f} | Train cost:{costs_train[best_epoch]:.3f} |"
    Val = f"|   Val loss:{valid_loss[best_epoch]:.3f} |   Val AUC:{AUCs_val[best_epoch]:.3f} |   Val cost:{costs_val[best_epoch]:.3f} |"
    Test = f"|  Test loss:{test_loss:.3f} |  Test AUC:{test_AUC:.3f} |  Test cost:{test_cost:.3f} |"
    print(summary)
    print(Train)
    print(Val)
    print(Test)
    file = open(file, "a")
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
