import torch
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
import os.path
import numpy as np

# import h5py

Image.MAX_IMAGE_PIXELS = None


def to_Tensor(path, name):
    """
    Load Tiff files as tensors
    """
    t = Image.open(path + "/" + name)
    t = ToTensor()(t)
    t = t.squeeze(dim=0)
    return t


def last_to_image(path, year):
    """
    Given path to folder having tiff files for each last band for given year
    returns Tensors with channels == bands and year as requested in the path
    """
    image = []
    for b in range(1, 5):
        band = Image.open(path + "/" + "_last_20%d_%d.tif" % (year, b))
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
    Creates categorical variables for deforestation event given cutoffs.
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
    end_year,
    tree_p=30,
    cutoffs=[2, 5, 8],
    sourcepath="/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/MadreTiff",
    rescale=True,
    wherepath=None,
):
    """
    Given year, and cutoffs as defined above returns (and save if wherepath!= None) 
        Static tensor,
        Non static tensor,
        list of valid pixels codrinates,
        list of labels corresponding to this valid cordinates
    
    sourcepath = path to tiff files
    wherepath = in not None, path to where to save the tensors
        
    Static tensor is identical for any year, hence save only once
    Static tensor has datamask layer and treecover
    
    Nonstatic tensor has if_deff_when cathegorical layers and the image landsat 7 bands stacked
    
    Valid pixels are these that meet all the following conditions :
     1. datamask == 1 , eg                        land not water body
     2. tree_cover > tree_p   or   gain == 1      if threecanpy in 2000 > tree_p or became forest up to 2012 
     3. lossyear > end_year   or   lossyear == 0  experienced loss only after that year (or not at all in the study period)    
     4. buffer == 0                               is in Madre de Dios area
     
    for each valid pixel asign label 1 if it is deforested in exactly in year+1 or zero otherwise  
    
    All pixels in the rasters and produced tensors have value 111 in the locations outside Madre de Dios and its buffer
    """
    buffer = to_Tensor(sourcepath, "if_in_buffer.tif")
    gain = to_Tensor(sourcepath, "gain_2018.tif")
    lossyear = to_Tensor(sourcepath, "lossyear_2018.tif")
    datamask = to_Tensor(sourcepath, "datamask_2018.tif")
    tree_cover = to_Tensor(sourcepath, "treecover2000_2018.tif")
    tree_cover = tree_cover.float()
    datamask = datamask.float()
    # Create list of valid pixels coordinates
    pixels = (
        (datamask == 1)
        & ((tree_cover > tree_p) | (gain == 1))  # land (not water body)
        & (  # if forest in 2000 or became forest up to 2012
            (lossyear > end_year) | (lossyear == 0)
        )
        & (  # experienced loss only after that year (or not at all in the study period)
            buffer == 0
        )
    ).nonzero()  # Madre de Dios Area

    # Create list of valid pixels labels in year + 1
    labels = lossyear[pixels[:, 0], pixels[:, 1]] == (
        end_year + 1
    )  # can be change to >= (end_year+1) & <111

    when = if_def_when(lossyear, end_year, cutoffs=cutoffs)
    image = last_to_image(sourcepath, end_year)

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
        torch.save(image, wherepath + "/" + "tensor_%d.pt" % (end_year))
        torch.save(pixels, wherepath + "/" + "pixels_cord_%d.pt" % (end_year))
        torch.save(labels, wherepath + "/" + "labels_%d.pt" % (end_year))

    return static, image, pixels, labels


class DatasetCNNHDF5(Dataset):
    """
    CNN Data class -- using HDF5 storage
        
        if it is passed list of image, pixels and labels, it concatenates the data as one, where inputs are
        all pairs of image:next year labels for valid pixels. Pairs are ordered as sequences in the same order as
        in the flatten list (NOT IMPLEMENTED YET)
        
        if list is of length 1, only one year pairs
        
        size is the radius of the image. Can be modified with Data.change_size(new size)   
    """

    def __init__(self, size, static, image, pixels, labels, indices):
        self.size = size
        self.lengths = None
        self.image = image
        self.pixels = pixels
        self.labels = torch.cat(labels, dim=0)
        self.static = static
        self.indices = indices

    def idx_to_image(self, idx):
        """
        given an index of a list of pixels in different years,
        return the corresponding image for the given year
        """
        return self.image[np.searchsorted(self.indices, idx, side="right") - 1]

    def change_size(self, new_size):
        self.size = new_size

    def __getitem__(self, idx):
        # todo: handle static
        image = self.idx_to_image(idx)
        image = torch.from_numpy(
            image[
                :,
                (self.pixels[idx, 0] - self.size) : (
                    self.pixels[idx, 0] + self.size + 1
                ),
                (self.pixels[idx, 1] - self.size) : (
                    self.pixels[idx, 1] + self.size + 1
                ),
            ]
        )

        static = self.static[
            :,
            (self.pixels[idx, 0] - self.size) : (self.pixels[idx, 0] + self.size + 1),
            (self.pixels[idx, 1] - self.size) : (self.pixels[idx, 1] + self.size + 1),
        ]
        label = self.labels[idx]
        cor = torch.from_numpy(self.pixels[idx])

        return torch.cat((static, image), dim=0), label, cor

    def __len__(self):
        return len(self.pixels)


class DatasetCNN(Dataset):
    """
    CNN Data class
        
        if it is passed list of image, pixels and labels, it concatenates the data as one, where inputs are
        all pairs of image:next year labels for valid pixels. Pairs are ordered as sequences in the same order as
        in the flatten list
        
        if list is of length 1, only one year pairs
        
        size is the radius of the image. Can be modified with Data.change_size(new size)   
    """

    def __init__(self, size, static, image, pixels, labels):
        self.size = size
        self.lengths = None

        if len(image) == 1:
            image = torch.cat(image, dim=0)
            self.image = torch.cat((static, image), dim=0)
            print("len(image) == 1")
        else:
            # add static to each image in the list so that all images are ready tensors
            # do this only when initializig the data class so that it is quick to call ready tensor at each get item call
            # save tensors in a list keeping the image order

            # save the lengths of each item in the pixles coordinates/labels list
            # so that after they are flattened, a map pixel,year -> image,year is possible
            self.lengths = [len(i) for i in pixels]
            self.image = []
            print("len(image) == %d" % len(image))
            for im in image:
                img = torch.cat((static, im), dim=0)
                self.image.append(img)

        self.pixels = torch.cat(pixels, dim=0)
        self.labels = torch.cat(labels, dim=0)

    def idx_to_image(self, idx):
        """
        given a index of a flatten list of pixels in different years,
        return the corresponding image for the given year
        """

        if self.lengths == None:
            image = self.image

        else:
            csum = list(np.cumsum(self.lengths))
            csum.insert(0, 0)
            for i in range(1, len(csum)):
                if (idx >= csum[i - 1]) & (idx < csum[i]):
                    image = self.image[i - 1]
                    break
        return image

    def change_size(self, new_size):
        self.size = new_size

    def __getitem__(self, idx):

        image = self.idx_to_image(idx)

        image = image[
            :,
            (self.pixels[idx, 0] - self.size) : (self.pixels[idx, 0] + self.size + 1),
            (self.pixels[idx, 1] - self.size) : (self.pixels[idx, 1] + self.size + 1),
        ]

        label = self.labels[idx]

        cor = self.pixels[idx]

        return image, label, cor

    def __len__(self):
        return len(self.pixels)


'''
def load_CNNdataHDF5(
    size,
    start_year,
    end_year,
    add_static=None,
    add_time=None,
    path="'/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/tensors'",
):
    """
    Not implemented for end_year > start_year yet!
    
    given start year, end year and size initialize CNN data class 
    start year and end year define how many pairs imange - next year label the data to have 
    size defines the returned image size
    path = path to saved tensors
    To add extra static layers, add_static must be a list of this tensors (2D or 3D for multi-channels)
    To add extra time layers, add_time must be a list of lists of this tensors (2D or 3D for multi-channels) where the
    lists are sorted in time and are of length end_year - start_year + 1
    """

    path = path + "/"
    static = torch.load(path + "static.pt")
    if add_static:
        for to_add in add_static:
            if len(to_add.shape) == 2:
                to_add = to_add.unsqueeze(dim=0)
                static = torch.cat([static, to_add], dim=0)
            else:
                static = torch.cat([static, to_add], dim=0)

    images_ls = []
    pixels_ls = []
    labels_ls = []

    for i, year in enumerate(range(start_year, end_year + 1)):
        print("i: %d, year: %d" % (i, year))

        image = h5py.File(path + "tensor_%d.hdf5" % year, mode="r")[
            "tensor"
        ]  # torch.load(path+"tensor_%d.pt"%(year))
        #        if(add_time):
        #            for to_add in add_time[i]:
        #                if len(to_add.shape) == 2 :
        #                    to_add = to_add.unsqueeze(dim = 0)
        #                    image = torch.cat([image,to_add], dim = 0)
        #                else:
        #                    image = torch.cat([image,to_add], dim = 0)

        images_ls.append(image)

        labels = torch.load(path + "labels_%d.pt" % (year))
        labels_ls.append(labels)

    pixelsH = h5py.File(path + "pixels_14_18.hdf5", mode="r")
    Data = DatasetCNNHDF5(
        size,
        static=static,
        image=images_ls,
        pixels=pixelsH["tensor"],
        labels=labels_ls,
        indices=pixelsH["indices"][:],
    )

    return Data
'''


def load_CNNdata(
    size,
    start_year,
    end_year,
    add_static=None,
    add_time=None,
    path="'/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/tensors'",
):
    """
    given start year, end year and size initialize CNN data class 
    start year and end year define how many pairs imange - next year label the data to have 
    size defines the returned image size
    path = path to saved tensors
    To add extra static layers, add_static must be a list of this tensors (2D or 3D for multi-channels)
    To add extra time layers, add_time must be a list of lists of this tensors (2D or 3D for multi-channels) where the
    lists are sorted in time and are of length end_year - start_year + 1
    """

    path = path + "/"
    static = torch.load(path + "static.pt")
    if add_static:
        for to_add in add_static:
            if len(to_add.shape) == 2:
                to_add = to_add.unsqueeze(dim=0)
                static = torch.cat([static, to_add], dim=0)
            else:
                static = torch.cat([static, to_add], dim=0)

    images_ls = []
    pixels_ls = []
    labels_ls = []

    for i, year in enumerate(range(start_year, end_year + 1)):
        print("i: %d, year: %d" % (i, year))
        image = torch.load(path + "tensor_%d.pt" % (year))
        if add_time:
            for to_add in add_time[i]:
                if len(to_add.shape) == 2:
                    to_add = to_add.unsqueeze(dim=0)
                    image = torch.cat([image, to_add], dim=0)
                else:
                    image = torch.cat([image, to_add], dim=0)

        images_ls.append(image)

        pixels = torch.load(path + "pixels_cord_%d.pt" % (year))
        pixels_ls.append(pixels)

        labels = torch.load(path + "labels_%d.pt" % (year))
        labels_ls.append(labels)

    Data = DatasetCNN(
        size, static=static, image=images_ls, pixels=pixels_ls, labels=labels_ls
    )

    return Data


# Regulate later if new layers are added in add_static or add_time
def with_DSM(size, start_year, end_year, wherepath, DSM=False, type="3D"):
    if type == "2D":
        if DSM:
            DSM = torch.load(wherepath + "/DSM.pt")
            Data = load_CNNdata(
                size=size,
                start_year=start_year,
                end_year=end_year,
                path=wherepath,
                add_static=[DSM],
            )
        else:
            Data = load_CNNdata(
                size=size,
                start_year=start_year,
                end_year=end_year,
                path=wherepath,
                add_static=None,
            )
    else:
        if DSM:
            print("load: " + wherepath + "/DSM.pt")
            DSM = torch.load(wherepath + "/DSM.pt")
            Data = load_RNNdata(
                size=size,
                start_year=start_year,
                end_year=end_year,
                path=wherepath,
                add_static=[DSM],
            )
        else:
            print("loading data")
            Data = load_RNNdata(
                size=size,
                start_year=start_year,
                end_year=end_year,
                path=wherepath,
                add_static=None,
            )

    return Data


class DatasetCNNforecast(Dataset):
    """
    CNN Data class of images and coordinates but no labels
    size is the radius of the mage. Can be modified with Data.change_size(new size)   
    """

    def __init__(self, size, static, image, pixels):
        self.size = size
        self.image = torch.cat((static, image), dim=0)
        self.pixels = pixels

    def change_size(self, new_size):
        self.size = new_size

    def __getitem__(self, idx):

        image = self.image[
            :,
            (self.pixels[idx, 0] - self.size) : (self.pixels[idx, 0] + self.size + 1),
            (self.pixels[idx, 1] - self.size) : (self.pixels[idx, 1] + self.size + 1),
        ]

        cor = self.pixels[idx]

        return image, cor

    def __len__(self):
        return len(self.pixels)


def load_CNNdata_forecast(
    size, year, path="'/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/tensors'"
):

    path = path + "/"
    DSM = torch.load(path + "DSM.pt")
    DSM = DSM.unsqueeze(dim=0)
    static = torch.load(path + "static.pt")
    static = torch.cat([static, DSM], dim=0)
    image = torch.load(path + "tensor_%d.pt" % (year))
    pixels = torch.load(path + "pixels_cord_%d.pt" % (year))

    Data = DatasetCNNforecast(size, static=static, image=image, pixels=pixels)

    return Data


class DatasetRNNforecast(Dataset):
    """
    RNN Data class of images and coordinates but no labels
    """

    def __init__(self, size, static, images, pixels):

        self.size = size
        self.static = static
        self.images = images
        self.pixels = pixels

    def change_size(self, new_size):
        self.size = new_size

    def __getitem__(self, idx):

        static = self.static[
            :,
            (self.pixels[idx, 0] - self.size) : (self.pixels[idx, 0] + self.size + 1),
            (self.pixels[idx, 1] - self.size) : (self.pixels[idx, 1] + self.size + 1),
        ]
        # (c x t x h x w)
        images = self.images[
            :,
            :,
            (self.pixels[idx, 0] - self.size) : (self.pixels[idx, 0] + self.size + 1),
            (self.pixels[idx, 1] - self.size) : (self.pixels[idx, 1] + self.size + 1),
        ]

        cor = self.pixels[idx]

        return (static, images), cor

    def __len__(self):
        return len(self.pixels)


class DatasetRNN(Dataset):
    """
    Data class for Models 2:4
    get_item return static tensor (to be fed in the static branch)
    and a 4d Tensor of non static images where the shape is as follows: 
    (c,t,h,w) = (channels per image ,time , h = 2*size+1, w = 2*size+1)
    change_size sets new image size: h&w = 2*new_size + 1
    """

    def __init__(self, size, static, images, pixels, labels):

        self.size = size
        self.static = static
        self.images = images
        self.pixels = pixels
        self.labels = labels

    def change_size(self, new_size):
        self.size = new_size

    def __getitem__(self, idx):

        static = self.static[
            :,
            (self.pixels[idx, 0] - self.size) : (self.pixels[idx, 0] + self.size + 1),
            (self.pixels[idx, 1] - self.size) : (self.pixels[idx, 1] + self.size + 1),
        ]
        # (c x t x h x w)
        images = self.images[
            :,
            :,
            (self.pixels[idx, 0] - self.size) : (self.pixels[idx, 0] + self.size + 1),
            (self.pixels[idx, 1] - self.size) : (self.pixels[idx, 1] + self.size + 1),
        ]
        label = self.labels[idx]

        cor = self.pixels[idx]

        return (static, images), label, cor

    def __len__(self):
        return len(self.pixels)


# THIS FUNCTION IS INCOMPLETE - DO NOT USE
def load_RNNdataHDF5(
    size,
    start_year,
    end_year,
    add_static=None,
    add_time=None,
    path="'/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/tensors'",
):
    """
    given start year, end year and size initilalize RNN data class BUT WITH HDF5
    start year and end year define number of elements in the time series of images
    size define the returned image size
    path = path to saved tensors
    To add extra static layers, than add_static must be a list of this tensors (2D or 3D for multi-channels)
    To add extra time layers, than add_time must be a list of lists of this tensors (2D or 3D for multi-channels) where the
    lists are sorted in time and are of lenght end_year - start_year + 1
    """
    path = path + "/"
    images = []
    for i, year in enumerate(range(start_year, end_year + 1)):

        image = torch.load(path + "tensor_%d.pt" % (year))

        if add_time:
            for to_add in add_time[i]:
                if len(to_add.shape) == 2:
                    to_add = to_add.unsqueeze(dim=0)
                    image = torch.cat([image, to_add], dim=0)
                else:
                    image = torch.cat([image, to_add], dim=0)

        image = image.unsqueeze(dim=1)
        images.append(image)

    images = torch.cat(images, dim=1)

    static = torch.load(path + "static.pt")

    if add_static:
        for to_add in add_static:
            if len(to_add.shape) == 2:
                to_add = to_add.unsqueeze(dim=0)
                static = torch.cat([static, to_add], dim=0)
            else:
                static = torch.cat([static, to_add], dim=0)

    pixels = torch.load(path + "pixels_cord_%d.pt" % (end_year))
    labels = torch.load(path + "labels_%d.pt" % (end_year))
    Data = DatasetRNN(
        size=size, images=images, static=static, pixels=pixels, labels=labels
    )
    return Data


def load_RNNdata(
    size,
    start_year,
    end_year,
    add_static=None,
    add_time=None,
    path="'/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/tensors'",
):
    """
    given start year, end year and size initilalize RNN data class 
    start year and end year define number of elements in the time series of images
    size define the returned image size
    path = path to saved tensors
    To add extra static layers, than add_static must be a list of this tensors (2D or 3D for multi-channels)
    To add extra time layers, than add_time must be a list of lists of this tensors (2D or 3D for multi-channels) where the
    lists are sorted in time and are of length end_year - start_year + 1
    """
    path = path + "/"
    images = []
    for i, year in enumerate(range(start_year, end_year + 1)):

        image = torch.load(path + "tensor_%d.pt" % (year))

        if add_time:
            for to_add in add_time[i]:
                if len(to_add.shape) == 2:
                    to_add = to_add.unsqueeze(dim=0)
                    image = torch.cat([image, to_add], dim=0)
                else:
                    image = torch.cat([image, to_add], dim=0)

        image = image.unsqueeze(dim=1)
        images.append(image)

    images = torch.cat(images, dim=1)

    static = torch.load(path + "static.pt")

    if add_static:
        for to_add in add_static:
            if len(to_add.shape) == 2:
                to_add = to_add.unsqueeze(dim=0)
                static = torch.cat([static, to_add], dim=0)
            else:
                static = torch.cat([static, to_add], dim=0)

    pixels = torch.load(path + "pixels_cord_%d.pt" % (end_year))
    labels = torch.load(path + "labels_%d.pt" % (end_year))
    Data = DatasetRNN(
        size=size, images=images, static=static, pixels=pixels, labels=labels
    )
    return Data


def load_RNNdata_forecast(
    size,
    start_year,
    end_year,
    add_static=None,
    add_time=None,
    path="/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/tensors",
):
    """
    given start year, end year and size initilalize RNN data class 
    start year and end year define number of elements in the time series of images
    size define the returned image size
    path = path to saved tensors
    To add extra static layers, than add_static must be a list of this tensors (2D or 3D for multi-channels)
    To add extra time layers, than add_time must be a list of lists of this tensors (2D or 3D for multi-channels) where the
    lists are sorted in time and are of length end_year - start_year + 1
    """
    path = path + "/"
    images = []
    for i, year in enumerate(range(start_year, end_year + 1)):

        image = torch.load(path + "tensor_%d.pt" % (year))

        if add_time:
            for to_add in add_time[i]:
                if len(to_add.shape) == 2:
                    to_add = to_add.unsqueeze(dim=0)
                    image = torch.cat([image, to_add], dim=0)
                else:
                    image = torch.cat([image, to_add], dim=0)

        image = image.unsqueeze(dim=1)
        images.append(image)

    images = torch.cat(images, dim=1)

    static = torch.load(path + "static.pt")

    if add_static:
        for to_add in add_static:
            if len(to_add.shape) == 2:
                to_add = to_add.unsqueeze(dim=0)
                static = torch.cat([static, to_add], dim=0)
            else:
                static = torch.cat([static, to_add], dim=0)

    pixels = torch.load(path + "pixels_cord_%d.pt" % (end_year))
    Data = DatasetRNNforecast(size=size, images=images, static=static, pixels=pixels)
    return Data


'''
# Doing all the data in one job can be beyond capacities, this function provides
# a way of splitting the data into managable chunks
def load_RNNdata_forecast2(
    size,
    start_year,
    end_year,
    add_static=None,
    add_time=None,
    path="/rdsgpfs/general/user/kpp15/home/Hansen/data/raster/tensors",
    splits,
    n,
):
    """
    given start year, end year and size initilalize RNN data class 
    start year and end year define number of elements in the time series of images
    size define the returned image size
    path = path to saved tensors
    To add extra static layers, than add_static must be a list of this tensors (2D or 3D for multi-channels)
    To add extra time layers, than add_time must be a list of lists of this tensors (2D or 3D for multi-channels) where the
    lists are sorted in time and are of length end_year - start_year + 1
    """
    path = path + "/"
    images = []
    for i, year in enumerate(range(start_year, end_year + 1)):

        image = torch.load(path + "tensor_%d.pt" % (year))
        x = np.array_split(range(image.shape[len(image.shape)-1]),splits)[n]
        image = image[:,:,x]

        if add_time:
            for to_add in add_time[i]:
                if len(to_add.shape) == 2:
                    to_add = to_add.unsqueeze(dim=0)
                    image = torch.cat([image, to_add], dim=0)
                else:
                    image = torch.cat([image, to_add], dim=0)

        image = image.unsqueeze(dim=1)
        x = np.array_split(range(image.shape[len(image.shape)-1]),splits)[n]
        image = image[:,:,x]
        images.append(image)

    images = torch.cat(images, dim=1)

    static = torch.load(path + "static.pt")

    if add_static:
        for to_add in add_static:
            if len(to_add.shape) == 2:
                to_add = to_add.unsqueeze(dim=0)
                static = torch.cat([static, to_add], dim=0)
            else:
                static = torch.cat([static, to_add], dim=0)

#    x = np.array_split(range(image.static[len(static.shape)-1]),splits)[n]
#    static = image[:,:,x]

    pixels = torch.load(path + "pixels_cord_%d.pt" % (end_year))
#    y = np.array_split(range(pixels.shape[0),splits)[n]
#    pixels = 
    Data = DatasetRNNforecast(size=size, images=images, static=static, pixels=pixels)
    return Data
'''

if __name__ == "__main__":

    import time

    server = "/rdsgpfs/general/user/kpp15/home/Hansen"
    sourcepath = server + "/data/raster/MadreTiff"
    wherepath = server + "/data/raster/tensors"
    end_year = 16
    start_year = 16
    size = 45

    since = time.time()
    DSM = torch.load(wherepath + "/DSM.pt")
    CNNdata = load_CNNdata(size, start_year, end_year, path=wherepath, add_static=[DSM])
    print("Time to load one year CNN data with DSM: ", time.time() - since)
    print("Image shape", CNNdata[0][0].shape)
    print("Data length", len(CNNdata))
    del CNNdata

    #     Time to load one year CNN data with DSM:  21.619445085525513
    #     Image shape torch.Size([11, 9, 9])
    #     Data length 107667735

    start_year = 14
    since = time.time()
    CNNdata = load_CNNdata(size, start_year, end_year, path=wherepath, add_static=[DSM])
    print("Time to load three years CNN data with DSM: ", time.time() - since)
    print("Image shape", CNNdata[0][0].shape)
    print("Data length", len(CNNdata))
    del CNNdata

    #     Time to load three years CNN data with DSM:  42.53541684150696
    #     Image shape torch.Size([11, 91, 91])
    #     Data length 324711556

    since = time.time()
    RNNdata = load_RNNdata(size, start_year, end_year, path=wherepath, add_static=[DSM])
    print("Time to load three years RNN data with DSM: ", time.time() - since)
    print("Static shape", RNNdata[0][0][0].shape)
    print("Image shape", RNNdata[0][0][1].shape)
    print("Data length", len(RNNdata))
    del RNNdata

#     Time to load three years RNN data with DSM:  44.5092191696167
#     Static shape torch.Size([3, 91, 91])
#     Image shape torch.Size([8, 3, 91, 91])
#     Data lenght 108005217

