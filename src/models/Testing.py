import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import scikitplot
from Training import AUC_CM
from scipy.special import expit as Sigmoid
import os
import time
from Data_maker_loader import *

# import rasterio as rio
import matplotlib
import matplotlib.pyplot as plt


def testing(
    model,
    Data,
    criterion,
    test_sampler,
    w,
    perc,
    batch_size,
    stop_batch,
    print_batch,
    name=None,
    path=None,
    save=False,
):
    print("\nTESTING ROUTINE COMMENCING")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    Test_sampler = SubsetRandomSampler(test_sampler.classIndexes_unsampled)
    test_loader = DataLoader(
        Data, sampler=Test_sampler, batch_size=batch_size, drop_last=True
    )
    require_sigmoid = isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss)

    # initialize lists to monitor test loss, accuracy and coordinates
    losses = []
    outputs = []
    targets = []
    coordinates = []

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
        losses.append(loss.item())
        outputs.append(list(output.cpu().data))
        targets.append(list(target.cpu().data))
        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        if print_batch:
            if batch % print_batch == 0:
                print("\tBatch:", batch, "\tLoss:", loss.item())
        # =====================================================================

    outputs = sum(outputs, [])
    targets = sum(targets, [])
    coordinates = sum(coordinates, [])
    t_time = time.time() - test_start
    print(
        "\n\tTime to load %d test batches of size %d : %3.4f hours\n"
        % (batch, batch_size, t_time / (3600))
    )

    losses = np.average(losses)

    print("\tTest Loss: ", losses)
    AUC, cost = AUC_CM(targets, outputs, w, perc, sigmoid=require_sigmoid)
    if require_sigmoid:
        outputs = np.array(Sigmoid(outputs))
    probas_per_class = np.stack((1 - outputs, outputs), axis=1)
    roc = scikitplot.metrics.plot_roc(np.array(targets), probas_per_class)

    coordinates = torch.stack(coordinates, dim=0)
    outputs = torch.tensor(outputs)

    if save:
        path = path + "/outputs"
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory ", path, " Created ")
        roc.get_figure().savefig(
            os.path.join(path, name + "_Test_ROC.png"), bbox_inches="tight"
        )
        torch.save(
            {"model_outputs": outputs, "targets": targets, "coordinates": coordinates},
            os.path.join(path, name + "_outputs.pt"),
        )
        print("saved at :", os.path.join(path, name + "_outputs.pt"))

    return outputs, targets, coordinates


def forecasting(
    model,
    Data,
    year,
    batch_size,
    stop_batch,
    print_batch,
    name=None,
    path=None,
    save=False,
):

    predict_loader = DataLoader(
        Data, shuffle=False, batch_size=batch_size, drop_last=False
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    outputs = []
    coordinates = []

    for batch, (data, cor) in enumerate(predict_loader):

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data, sigmoid=True)
        outputs.append(list(output.cpu().data))
        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        if print_batch:
            if batch % print_batch == 0:
                print("\tBatch:", batch)
        # =====================================================================

    outputs = sum(outputs, [])
    coordinates = sum(coordinates, [])
    coordinates = torch.stack(coordinates, dim=0)
    outputs = torch.tensor(outputs)

    if save:
        path = path + "/outputs"
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory ", path, " Created ")
        torch.save(
            {"model_outputs": outputs, "coordinates": coordinates},
            os.path.join(path, name + "_forecast_outputs_from%d.pt" % (year)),
        )
        print(
            "saved at :",
            os.path.join(path, name + "_forecast_outputs_from%d.pt" % (year)),
        )

    return outputs, coordinates


def forecasting_split(
    model,
    Data,
    year,
    batch_size,
    stop_batch,
    print_batch,
    name=None,
    path=None,
    save=False,
    slice=1,
):

    predict_loader = DataLoader(
        Data, shuffle=False, batch_size=batch_size, drop_last=False
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    outputs = []
    coordinates = []

    for batch, (data, cor) in enumerate(predict_loader):

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data, sigmoid=True)
        outputs.append(list(output.cpu().data))
        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        if print_batch:
            if batch % print_batch == 0:
                print("\tBatch:", batch)
        # =====================================================================

    outputs = sum(outputs, [])
    coordinates = sum(coordinates, [])
    coordinates = torch.stack(coordinates, dim=0)
    outputs = torch.tensor(outputs)

    if save:
        path = path + "/forecasts"
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory ", path, " Created ")
        torch.save(
            {"model_outputs": outputs, "coordinates": coordinates},
            os.path.join(
                path, name + "Madre_forecast_outputs_from%d_%d.pt" % (year, slice)
            ),
        )
        print(
            "saved at :",
            os.path.join(
                path, name + "Madre_forecast_outputs_from%d_%d.pt" % (year, slice)
            ),
        )
        np.savetxt(
            os.path.join(
                path, name + "Madre_forecast_outputs_from%d_%d" % (year, slice) + ".csv"
            ),
            np.c_[coordinates.numpy(), outputs.numpy()],
            delimiter=",",
        )
        print(
            "saved: ",
            name + "Madre_forecast_outputs_from%d_%d" % (year, slice) + ".csv",
        )

    return outputs, coordinates


"""
def heatmap(
    end_year, outputs, coordinates, sourcepath, wherepath, savepath, name, msk_yr=19
):

    datamask = datamask = to_Tensor(sourcepath, "datamask_20%d.tif" % (msk_yr))
    valid_pixels = torch.load(wherepath + "/pixels_cord_%d.pt" % (end_year))
    datamask[valid_pixels[:, 0], valid_pixels[:, 1]] = 0

    print("Valid pixels to predict in year 20%d" % (end_year))
    colors = ["white", "green", "grey", "blue"]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.matshow(datamask, cmap=matplotlib.colors.ListedColormap(colors))
    fig.show()

    heatmap = torch.ones(datamask.shape) * (-1)
    heatmap[coordinates[:, 0], coordinates[:, 1]] = outputs
    heatmap = heatmap.numpy()
    heatmap[heatmap == -1] = None

    print("Heatmap:")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.matshow(
        heatmap, cmap=matplotlib.cm.Spectral_r, interpolation="none", vmin=0, vmax=1
    )
    fig.show()

    with rio.open(sourcepath + "datamask_20%d.tif" % (msk_yr)) as src:
        ras_data = src.read()
        ras_meta = src.profile

    # make any necessary changes to raster properties, e.g.:
    # is this the correct way - should us .update?
    ras_meta["dtype"] = "float32"
    ras_meta["nodata"] = -1

    # where to save the output .tif
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        print("Directory ", savepath, " Created ")

    with rio.open(savepath + name + f"_20{end_year:d}.tif", "w", **ras_meta) as dst:
        dst.write(heatmap, 1)

    print(
        "Heatmap min: %.4f, max: %.4f, mean: %.4f;"
        % (np.nanmin(heatmap), np.nanmax(heatmap), np.nanmean(heatmap))
    )
    print("heatmap saved at: ", savepath + name + f"_20{end_year:d}.tif")
"""

# For saving as .csv batches
# Seems incomplete to me
def forecasting2(
    model, Data, year, batch_size, stop_batch, print_batch, name=None, path=None
):

    predict_loader = DataLoader(Data, shuffle=True, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    outputs = []
    coordinates = []

    for batch, (data, cor) in enumerate(predict_loader):

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data, sigmoid=True)
        #        outputs.append(list(output.cpu().data))
        #        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        print("\tBatch:", batch)
        np.savetxt(
            path + str(batch) + "batch.csv",
            np.c_[cor.cpu().data, output.cpu().data],
            delimiter=",",
        )
        print("saved: ", path + str(batch) + "batch.csv")
        # =====================================================================
