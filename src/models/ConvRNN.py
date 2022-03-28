import torch
from spp_layer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn

# Initially from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
# Updated at https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # does 4 represent years
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        #         return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
        #                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        return (
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width).data,
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width).data,
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_size=(21, 21),
        input_dim=5,
        hidden_dim=(16, 32),
        kernel_size=((3, 3),),
        num_layers=2,
        bias=True,
        return_all_layers=False,
    ):

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        if not len(kernel_size) == num_layers:
            kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        if not len(hidden_dim) == num_layers:
            hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.height, self.width = input_size
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_size=self.input_size,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        Returns
        -------
        last_state_list, layer_output
        """

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        #         layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(2)  # Number of years worth of dynamic tensors
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, :, t, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=2)
            cur_layer_input = layer_output

            #             returns all [layer_1(h_1,h_2,...h_t),layer_2(h_1,h_2,...h_t),layer_3(h_1,h_2,...h_t)...]
            #             dont need it if not tracking individual loss
            #             layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            #             layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        #          return layer_output_list, last_state_list
        return last_state_list[0]

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param[0]] * num_layers
        return param


# Adapted from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/sequenceencoder.py
class LSTMSequentialEncoder(torch.nn.Module):
    def __init__(
        self,
        height=21,
        width=21,
        input_dim=(2, 5),
        hidden_dim=(16, 16, 64, 8),
        kernel_size=((3, 3), (1, 3, 3), (3, 3), (3, 3)),
        levels=(13,),
        dropout=0.2,
        bias=True,
    ):
        super(LSTMSequentialEncoder, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
        )

        self.inconv = nn.Sequential(
            torch.nn.Conv3d(input_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
        )

        cell_input_size = height - 3 * (kernel_size[1][-1] - 1)
        self.cell = ConvLSTMCell(
            input_size=(cell_input_size, cell_input_size),
            input_dim=hidden_dim[1],
            hidden_dim=hidden_dim[2],
            kernel_size=kernel_size[2],
            bias=bias,
        )

        self.final = nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[2] + hidden_dim[0], hidden_dim[3], kernel_size[3]
            ),
            torch.nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[3]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[3] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):
        # Split into static (z) and dynamic tensors (x) to be fed into different branches
        z, x = data
        # 2D convolutions over the static tensor
        z = self.conv.forward(z)
        #
        x = self.inconv.forward(x)
        # bands, channels, time, height, width
        b, c, t, h, w = x.shape
        hidden = torch.zeros((b, self.hidden_dim[2], h, w))
        state = torch.zeros((b, self.hidden_dim[2], h, w))
        for iter in range(t):
            hidden, state = self.cell.forward(x[:, :, iter, :, :], (hidden, state))
        x = hidden
        # Join dynamic and static branches
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()


class DeepLSTMSequentialEncoder(torch.nn.Module):
    """
    DeepLSTMSequentialEncoder with the option to add multiple ConvLSTM layers
    """

    def __init__(
        self,
        height=21,
        width=21,
        input_dim=(2, 5),
        hidden_dim=(16, 16, (16, 16), 8),
        kernel_size=((3, 3), (1, 3, 3), ((3, 3),), (3, 3)),
        num_layers=2,
        levels=(13,),
        dropout=0.2,
        bias=True,
        return_all_layers=False,
    ):
        super(DeepLSTMSequentialEncoder, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
        )

        self.inconv = nn.Sequential(
            torch.nn.Conv3d(input_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
        )

        cell_input_size = height - 3 * (kernel_size[1][1] - 1)

        self.cell = ConvLSTM(
            input_size=(cell_input_size, cell_input_size),
            input_dim=hidden_dim[1],
            hidden_dim=hidden_dim[2],
            kernel_size=kernel_size[2],
            num_layers=num_layers,
            bias=bias,
            return_all_layers=return_all_layers,
        )

        self.final = nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[2][-1] + hidden_dim[0], hidden_dim[3], kernel_size[3]
            ),
            torch.nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[3]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[3] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        z = self.conv.forward(z)
        x = self.inconv.forward(x)
        hidden, state = self.cell.forward(x)
        x = hidden
        # Join dynamic and static branches
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)
        return x.flatten()


class Conv_3D(torch.nn.Module):
    """
    Making deforestation predictions with 3D convolutions (space + time)
    """

    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
        start_year=14,
        end_year=17,
    ):
        super(Conv_3D, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            # This second 3d conv layer is troublesome
            # Kernel size needs to be tweaked by year
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=(
                    kernel_size[1][0] + (end_year - start_year - 2),
                    kernel_size[1][1],
                    kernel_size[1][2],
                ),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = x.squeeze(dim=2)
        # print("x shape post squeeze:", x.shape)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()


# Kernel size needs to be different depending on how many years of data are being handled
# This model is for an even number of training years (e.g. start_date = 14, end_date = 17)
class Conv_3Deven(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3Deven, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                # This one for odd num of years#
                # kernel_size = kernel_size[1]),
                # This one for even num of years#
                kernel_size=(
                    kernel_size[1][0] + 1,
                    kernel_size[1][1],
                    kernel_size[1][2],
                ),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        # print("z shape start:", z.shape)
        # print("x shape start:", x.shape)
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        # print("z shape post conv2d:", z.shape)
        # print("x shape post conv3d:", x.shape)
        x = x.squeeze(dim=2)
        # print("x shape post squeeze:", x.shape)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()


# Kernel size needs to be different depending on how many years of data are being handled
# This model is for an odd number of training years (e.g. start_date = 14, end_date = 16)
class Conv_3Dodd(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3Dodd, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                # This one for odd num of years#
                # kernel_size=kernel_size[1],
                # This one for even num of years#
                kernel_size=(
                    kernel_size[1][0] + 2,
                    kernel_size[1][1],
                    kernel_size[1][2],
                )
                # ),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        print("z shape start:", z.shape)
        print("x shape start:", x.shape)
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        print("z shape post conv2d:", z.shape)
        print("x shape post conv3d:", x.shape)
        x = x.squeeze(dim=2)
        print("x shape post squeeze:", x.shape)
        x = torch.cat((x, z), dim=1)  # Problem with dimensions here
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()


# Updated to change how labels are handled - 2 labels instead of one
class Conv_3DoddT(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3DoddT, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                # This one for odd num of years#
                kernel_size=kernel_size[1],
            ),
            # This one for even num of years#
            #                                        kernel_size = (kernel_size[1][0]+1,kernel_size[1][1],kernel_size[1][2])),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 2),
        )  # changed to 2

    #        self.sig = torch.nn.Sigmoid()

    #        self.sfmx = torch.nn.Softmax(dim=1)

    def forward(self, data, sigmoid=True):

        z, x = data

        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = x.squeeze(dim=2)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)

        #        if sigmoid:
        #            x = self.sig(x)
        #        x = self.sfmx(x) # need this?

        return x


# Model graveyard
# class Conv_3D(torch.nn.Module):
#     def __init__(self, input_dim=(2,5),
#                  hidden_dim=(16,16,64),
#                  kernel_size=((3,3),(2,3,3),(3,3)),
#                  levels=(12,),
#                  dropout = 0.2):
#         super(Conv_3D, self).__init__()

#         self.levels = levels
#         self.hidden_dim = hidden_dim

#         self.conv_2D = nn.Sequential(
#             nn.Conv2d(input_dim[0],hidden_dim[0],kernel_size = kernel_size[0]),
#             nn.ReLU(),
#             nn.BatchNorm2d(hidden_dim[0]),

#             nn.Conv2d(hidden_dim[0],hidden_dim[0],kernel_size = kernel_size[0]),
#             nn.ReLU(),
#             nn.BatchNorm2d(hidden_dim[0]))

#         self.conv_3D = nn.Sequential(
#                         torch.nn.Conv3d(in_channels = input_dim[1],
#                                         out_channels = hidden_dim[1],
#                                         kernel_size = kernel_size[1]),
#                         nn.ReLU(),
#                         nn.BatchNorm3d(hidden_dim[1]),

#                         torch.nn.Conv3d(in_channels = hidden_dim[1],
#                                         out_channels = hidden_dim[1],
#                                         kernel_size = kernel_size[1]),
#                         nn.ReLU(),
#                         nn.BatchNorm3d(hidden_dim[1]))

#         self.final = nn.Sequential(
#                         torch.nn.Conv2d(hidden_dim[0]+hidden_dim[1], hidden_dim[2], kernel_size[2]),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(hidden_dim[2]),

#                         torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(hidden_dim[2]),

#                         torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(hidden_dim[2]),

#                         torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(hidden_dim[2]))

#         ln_in = 0
#         for i in levels:
#             ln_in += hidden_dim[2]*i*i

#         self.ln = torch.nn.Sequential(
#             torch.nn.Linear(ln_in,100),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(100),
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(100, 1))

#         self.sig = torch.nn.Sigmoid()


#     def forward(self, data , sigmoid = True ):

#         z , x = data

#         z = self.conv_2D.forward(z)
#         x = self.conv_3D.forward(x)
#         x = x.squeeze(dim = 2 )
#         x = torch.cat((x,z),dim = 1)
#         print("Before final CNN: ",x.shape)
#         x = self.final.forward(x)
#         print("After final CNN: ",x.shape)
#         x = spp_layer(x, self.levels)
# #         print(x.shape)
#         x= self.ln(x)
# #         print(x.shape)
#         if sigmoid:
#             x = self.sig(x)

#         return x.flatten()
