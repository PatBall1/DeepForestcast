from torchvision.transforms import ToTensor
from PIL import Image
#import rasterio
import numpy as np
import torch
import os.path


def to_Tensor(path, name):
    """
    Load Tiff files as tensors
    """
    t = Image.open(path + "/" + name)
    t = ToTensor()(t)
    t = t.squeeze(dim=0)
    return t

'''
def to_Tensor(path, name):
    """
    Load Tiff files as tensors
    """
    t = rasterio.open(path + "/" + name)
    t = ToTensor()(t)
    t = t.squeeze(dim=0)
    return t
'''
def last_to_image(path, year):
    """
    Given path to folder having tiff files for each last band for given year
    returns Tensors with chanels == bands and year as requested in the path
    """
    image = []
    for b in range(1, 5):
        band = Image.open(path + "/last_20%d_%d.tif" % (year, b))
        band = ToTensor()(band)
        image.append(band)
    image = torch.cat(image, dim=0)
    image = image.float()
    return image


def rescale_image(image):
    # detach and clone the image so that you don't modify the input, but are returning new tensor.
    rescaled_image = image.data.clone()
    if len(image.shape) == 2:
        rescaled_image = rescaled_image.unsqueeze(dim=0)
    # Compute mean and std only from non masked pixels
    # Spatial coordinates of this pixels are:
    pixels = (rescaled_image[0, :, :] != -1).nonzero()
    mean = rescaled_image[:, pixels[:, 0], pixels[:, 1]].mean(1, keepdim=True)
    std = rescaled_image[:, pixels[:, 0], pixels[:, 1]].std(1, keepdim=True)
    rescaled_image[:, pixels[:, 0], pixels[:, 1]] -= mean
    rescaled_image[:, pixels[:, 0], pixels[:, 1]] /= std
    if len(image.shape) == 2:
        rescaled_image = rescaled_image.squeeze(dim=0)
        mean = mean.squeeze(dim=0)
        std = std.squeeze(dim=0)
    return (rescaled_image, mean, std)


def if_def_when(lossyear, year, cutoffs=[2, 5, 8]):
    """
    Creates categorical variables for deforestration event given cutoffs.
    Values in cutoffs define the time bins
    Returns len(cutoffs) + 1 categorical layers:
    Example: cutoffs = [2,5,8], num of layers = 4 , considered year = year
    Categories: 
    0) if year - lossyear is in [0,2) 
    1) if year - lossyear is in [2,5) 
    2) if year - lossyear is in [5,8) 
    3) 8 years ago or more
    No prior knowledge:
        if loss event is in year > considered year or pixel is non deforested up to 2018+, all categories have value 0 
    """
    cutoffs.append(year)
    cutoffs.insert(0, 0)
    lossyear[(lossyear > year)] = 0
    losses = []
    for idx in range(0, len(cutoffs) - 1):
        deff = torch.zeros(lossyear.size())
        deff[
            (cutoffs[idx] <= (year - lossyear)) & ((year - lossyear) < cutoffs[idx + 1])
        ] = 1
        losses.append(deff.float())
    losses = torch.stack(losses)
    # Return Nan values encoded as needed:
    losses[:, (lossyear == -1)] = -1
    return losses


def create_tnsors_pixels(
    year,
    latest_yr,
    tree_p=30,
    cutoffs=[2, 5, 8],
    sourcepath=None,
    rescale=True,
    wherepath=None,
):
    """
    Given year, and cutoffs as defined above returns (and save if wherepath!= None) 
        Static tensor,
        Non static tensor,
        list of valid pixels coordinates,
        list of labels corresponding to this valid cordinates
    
    sourcepath = path to tiff files
    wherepath = in not None, path to where to save the tensors
        
    Static tensor is identical for any year, hence save only once
    Static tensor has datamask layer and treecover
    
    Nonstatic tensor has if_deff_when categorical layers and the image landset 7 bands stacked
    
    Valid pixels are these that meet all the following conditions :
     1. datamask == 1 , eg                        land not water body
     2. tree_cover > tree_p   or   gain == 1      if tree canopy in 2000 > tree_p or became forest up to 2012 
     3. lossyear > year   or   lossyear == 0      experienced loss only after that year (or not at all in the study period)    
     4. buffer == 0                               is in Madre de Dios area
     
    for each valid pixel assign label 1 if it is deforested in exactly in year+1 or zero otherwise  
    
    All pixels in the rasters and produced tensors have value 111 in the locations outside Area of Interest and its buffer
    """
    buffer = to_Tensor(sourcepath, "buffer.tif")
    gain = to_Tensor(sourcepath, "gain_20" + str(latest_yr) + ".tif")
    lossyear = to_Tensor(sourcepath, "lossyear_20" + str(latest_yr) + ".tif")
    datamask = to_Tensor(sourcepath, "datamask_20" + str(latest_yr) + ".tif")
    tree_cover = to_Tensor(sourcepath, "treecover2000_20" + str(latest_yr) + ".tif")
    tree_cover = tree_cover.float()
    datamask = datamask.float()
    # Create list of valid pixels coordinates
    pixels = (
        (datamask == 1)
        & ((tree_cover > tree_p) | (gain == 1))  # land (not water body)
        & (  # if forest in 2000 or became forest up to 2012
            (lossyear > year) | (lossyear == 0)
        )
        & (  # experienced loss only after that year (or not at all in the study period)
            buffer == 0
        )
    ).nonzero()  # In area of interest

    # Create list of valid pixels labels in year + 1
    labels = lossyear[pixels[:, 0], pixels[:, 1]] == (
        year + 1
    )  # can be change to >= (year+1) & <111
    # Could add here labels for +2 years
    #labels2 = lossyear[pixels[:, 0], pixels[:, 1]] == (
    #    year + 2
    #)
    #labels = labels + 2*labels2
    when = if_def_when(lossyear, year, cutoffs=cutoffs)
    image = last_to_image(sourcepath, year)

    if rescale:
        # Rescale datamask to have values -1 for nan, 0 for land, 1 for water
        datamask[datamask != -1] = datamask[datamask != -1] - 1
        # Rescale tree_cover to have values in [0, 1] and -1 for nan
        tree_cover[tree_cover != -1] = tree_cover[tree_cover != -1] * 0.01
        # Normalize image by channel with -1 values for nan
        image, _, _ = rescale_image(image)

    # Create non Static tensor
    image = torch.cat((when, image), dim=0)
    # Creates static tensor
    static = torch.stack((datamask, tree_cover))

    # Creates non static tensor
    if wherepath:
        if not os.path.isfile(wherepath + "/" + "static.pt"):
            torch.save(static, wherepath + "/" + "static.pt")
        torch.save(image, wherepath + "/" + "tensor_%d.pt" % (year))
        torch.save(pixels, wherepath + "/" + "pixels_cord_%d.pt" % (year))
        torch.save(labels, wherepath + "/" + "labels_%d.pt" % (year))

    return static, image, pixels, labels

