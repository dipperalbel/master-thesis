import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet
from .utils.channel_shuffle import Shuffle

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        """
        Args:
            dim:
            s:
        """
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        """
        Args:
            x:
        """
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):

        """Initializes Unit3D module.

        Args:
            in_channels:
            output_channels:
            kernel_shape:
            stride:
            padding:
            activation_fn:
            use_batch_norm:
            use_bias:
            name:
        """
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            # We always want padding to be 0 here.
            # We will dynamically pad based on input size in forward function
            padding=0,
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        """
        Args:
            dim:
            s:
        """
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        """
        Args:
            x:
        """
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        """
        Args:
            in_channels:
            out_channels:
            name:
        """
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[5, 5, 5],
            name=name + "/Branch_1/Conv3d_0b_5x5",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b2c = Unit3D(
            in_channels=out_channels[4],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_1b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x):
        """
        Args:
            x:
        """
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2c(self.b2b(self.b2a(x)))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class I3D_Shufflenet(nn.Module):
    """Inception-v1 I3D architecture. The model is introduced in:

        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset Joao
        Carreira, Andrew Zisserman https://arxiv.org/pdf/1705.07750v1.pdf.

    See also the Inception architecture, introduced in:
        Going deeper with convolutions Christian Szegedy, Wei Liu, Yangqing Jia,
        Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent
        Vanhoucke, Andrew Rabinovich. http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        spatial_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_class=6
    ):
        """Initializes I3D model instance. :param spatial_squeeze: Whether to
        squeeze the spatial dimensions for the logits

            before returning (default True).

        Args:
            spatial_squeeze:
            final_endpoint: The model contains many possible endpoints.
                `final_endpoint` specifies the last endpoint for the model to be
                built up to. In addition to the output at `final_endpoint` , all
                the outputs at endpoints up to `final_endpoint` will also be
                returned, in a dictionary. `final_endpoint` must be one of
                InceptionI3d.VALID_ENDPOINTS (default 'Logits').
            name: A string (optional). The name of this module.
            in_channels:
            dropout_keep_prob:

        Raises:
            ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % final_endpoint)

        super(I3D_Shufflenet, self).__init__()
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % self._final_endpoint)

        self.end_points = nn.ModuleDict({})
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "shuffle"
        self.end_points[end_point] = Shuffle(int(528/2), 528, 988)

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_5a"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 21, 128, 128], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [348, 192, 384, 48, 128, 128], name + end_point
        )
        if self._final_endpoint == end_point:
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.build()
            return
        
        end_point = "AvgPool_5e"
        self.end_points[end_point] = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))

        end_point = "Logits"
        self.end_points[end_point] = Unit3D(
            in_channels=988,
            output_channels=num_class,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
            activation_fn=None, 
            use_bias=False,
            use_batch_norm=False
        )


        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        """
        Args:
            x:
        """
        x = self.end_points['Conv3d_1a_7x7'](x)
        x = self.end_points['MaxPool3d_2a_3x3'](x)

        x = self.end_points['Conv3d_2b_1x1'](x)
        x = self.end_points['Conv3d_2c_3x3'](x)
        x = self.end_points['MaxPool3d_3a_3x3'](x)

        x = self.end_points['Mixed_3b'](x)
        x = self.end_points['Mixed_3c'](x)
        x = self.end_points['MaxPool3d_4a_3x3'](x)

        x = self.end_points['Mixed_4b'](x)
        x = self.end_points['Mixed_4c'](x)
        x = self.end_points['Mixed_4d'](x)
        x = self.end_points['Mixed_4e'](x)

        s = self.end_points['shuffle'](x)

        x = self.end_points['MaxPool3d_5a_2x2'](x)

        x = self.end_points['Mixed_5a'](x)
        x = self.end_points['Mixed_5b'](x)
        x = self.end_points['Mixed_5c'](x)

        x = (x + s)/2

        x = self.end_points['AvgPool_5e'](x)
        x = self.end_points['Logits'](x).squeeze(dim=3).squeeze(dim=3)

        x = torch.mean(x, dim=2)
        return x




def load_i3d_imagenet_pretrained():
    module_convertion_table = {
        "conv1": "Conv3d_1a_7x7",
        "maxpool1": "MaxPool3d_2a_3x3",
        "conv2": "Conv3d_2b_1x1",
        "conv3": "Conv3d_2c_3x3",
        "maxpool2": "MaxPool3d_3a_3x3",
        "inception3a": "Mixed_3b",
        "inception3b": "Mixed_3c",
        "maxpool3": "MaxPool3d_4a_3x3",
        "inception4a": "Mixed_4b",
        "inception4b": "Mixed_4c",
        "inception4c": "Mixed_4d",
        "inception4d": "Mixed_4e",
        "inception4e": "Mixed_4f",
        "maxpool4": "MaxPool3d_5a_2x2",
        "inception5a": "Mixed_5b",
        "inception5b": "Mixed_5c",
    }

    branch_convertion_table = {
        "branch1": "b0",
        "branch2.0": "b1a",
        "branch2.1": "b1b",
        "branch3.0": "b2a",
        "branch3.1": "b2b",
        "branch4.0": "b3a",
        "branch4.1": "b3b",
    }

    weights = I3D_Shufflenet().state_dict()

    pretrained_weights = googlenet(pretrained=True).state_dict()
    for k, data in pretrained_weights.items():
        # convert all names
        for orig, convertion in module_convertion_table.items():
            k = k.replace(orig, convertion)
        for orig, convertion in branch_convertion_table.items():
            k = k.replace(orig, convertion)
        k = k.replace(".conv.", ".conv3d.")

        if "conv" in k:
            # weights of size channels_out, channels_in, h, w
            data = data.unsqueeze(2)
            # weights now have a dummy T dimension (channels_out, channels_in, T, h, w)
            h = data.size(3)
            # expand dimensions
            data = data.repeat(1, 1, h, 1, 1)

            weights[k].data.copy_(data)
        elif "bn" in k:
            weights[k].data.copy_(data)
        elif "fc" in k:
            pass
        else:
            raise ValueError("problem with key", k)

    return weights

if __name__ == '__main__':
    inc = I3D_Shufflenet()
    a = torch.zeros([10, 3, 64, 224, 224])
    out = inc(a)
    print(out.shape)