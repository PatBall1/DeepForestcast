import torch
import numpy as np


def spp_layer(input_, levels=[6]):
    shape = input_.shape[-3:]
    pyramid = []
    for n in levels:
        if (n > shape[1]) or (n > shape[2]):
            raise ("Levels are not correct!")
        else:
            stride_1 = np.floor(float(shape[1] / n)).astype(np.int32)
            stride_2 = np.floor(float(shape[2] / n)).astype(np.int32)
            ksize_1 = stride_1 + (shape[1] % n)
            ksize_2 = stride_2 + (shape[2] % n)
            pool_layer = torch.nn.MaxPool2d(
                kernel_size=(ksize_1, ksize_2), stride=(stride_1, stride_2)
            )
            pool = pool_layer(input_)
            pool = pool.view(-1, shape[0] * n * n)
        pyramid.append(pool)
    spp_pool = torch.cat(pyramid, 1)
    return spp_pool


if __name__ == "__main__":
    levels = [15, 4, 2]
    in_size = 20
    x = torch.rand([2, 10, in_size, in_size])
    print(spp_layer(x, levels).shape)

    in_size = 15
    x = torch.rand([2, 10, in_size, in_size])
    print(spp_layer(x, levels).shape)

    in_size = 10
    x = torch.rand([2, 10, in_size, in_size])
    print(spp_layer(x, levels).shape)
