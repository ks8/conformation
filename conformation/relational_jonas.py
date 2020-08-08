""" Improved relational architecture. """
import copy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


# noinspection PyPep8Naming
def triu_indices_flat(N, k=0):
    """

    :param N:
    :param k:
    :return:
    """
    return np.argwhere(np.triu(np.ones(N, dtype=np.uint8), k=k).flatten()).flatten()


def goodmax(x, dim):
    """

    :param x:
    :param dim:
    :return:
    """
    return torch.max(x, dim=dim)[0]


class MaskedBatchNorm1d(nn.Module):
    """
    Stuff.
    """

    def __init__(self, feature_n):
        """
        Batchnorm1d that skips some rows in the batch
        """

        super(MaskedBatchNorm1d, self).__init__()
        self.feature_n = feature_n
        self.bn = nn.BatchNorm1d(feature_n)

    def forward(self, x, mask):
        """

        :param x:
        :param mask:
        :return:
        """
        assert x.shape[0] == mask.shape[0]
        assert mask.dim() == 1

        bin_mask = mask > 0
        y_i = self.bn(x[bin_mask])
        y = torch.zeros(x.shape, device=x.device)
        y[bin_mask] = y_i
        return y


# noinspection PyPep8Naming
def batch_mat_chan_eye(BATCH_N, MAT_N, CHAN_N):
    """

    :param BATCH_N:
    :param MAT_N:
    :param CHAN_N:
    :return:
    """
    A = torch.eye(MAT_N)
    B = A.repeat(BATCH_N, CHAN_N, 1, 1).transpose(1, -1)
    return B


# noinspection PyPep8Naming
def batch_mat_chan_norm(A, add_identity=True, reg_eps=1e-4):
    """
    A is BATCHN x MAT_N x MAT_N x C
    and we normalize each (MAT_N, MAT_N) matrix

    Adding identity adds the identity first, this also
    regularizes the 1/sqrt
    """

    if add_identity:
        # noinspection PyAugmentAssignment
        A = A + batch_mat_chan_eye(A.shape[0], A.shape[1], A.shape[-1]).to(A.device)
        reg_eps = 0

    s = 1.0 / torch.sqrt(A.sum(dim=-2) + reg_eps)

    # noinspection PyRedundantParentheses
    return (s.unsqueeze(-2) * A * s.unsqueeze(-3))


# noinspection PyPep8Naming,PyShadowingNames
def create_mp(MAX_N, F, STEP_N, args):
    """

    :param MAX_N:
    :param F:
    :param STEP_N:
    :param args:
    :return:
    """
    args = copy.deepcopy(args)
    class_name = args['name']
    del args['name']
    return eval(class_name)(MAX_N=MAX_N, F=F,
                            STEP_N=STEP_N, **args)


class Vpack:
    """
    Stuff.
    """

    # noinspection PyPep8Naming,PyShadowingNames
    def __init__(self, BATCH_N, MAX_N, F, mask):
        """

        """
        self.BATCH_N = BATCH_N
        self.F = F
        self.MAX_N = MAX_N
        self.mask = mask

    # noinspection PyPep8Naming
    def zero(self, V):
        """

        :param V:
        :return:
        """
        mask = self.mask.reshape(-1).unsqueeze(-1)
        return (V.reshape(-1, self.F) * mask).reshape(V.shape)

    # noinspection PyPep8Naming
    def pack(self, V):
        """

        :param V:
        :return:
        """
        # noinspection PyPep8Naming
        V_flat = V.reshape(-1, self.F)
        mask = (self.mask > 0).reshape(-1)
        return V_flat[mask]

    # noinspection PyPep8Naming
    def unpack(self, V):
        """

        :param V:
        :return:
        """
        output = torch.zeros((self.BATCH_N * self.MAX_N, V.shape[-1]), device=V.device)
        mask = (self.mask > 0).reshape(-1)
        output[mask] = V
        return output.reshape(self.BATCH_N, self.MAX_N, V.shape[-1])


class Epack:
    """
    Stuff.
    """

    # noinspection PyPep8Naming,PyShadowingNames
    def __init__(self, BATCH_N, MAX_N, F, mask):
        """

        """
        self.BATCH_N = BATCH_N
        self.F = F
        self.MAX_N = MAX_N
        self.mask = mask

    # noinspection PyPep8Naming
    def zero(self, E):
        """

        :param E:
        :return:
        """
        mask = self.mask.reshape(-1).unsqueeze(-1)
        return (E.reshape(-1, self.F) * mask).reshape(E.shape)

    # noinspection PyPep8Naming
    def pack(self, E):
        """

        :param E:
        :return:
        """
        # noinspection PyPep8Naming
        E_flat = E.reshape(-1, self.F)
        mask = (self.mask > 0).reshape(-1)
        return E_flat[mask]

    # noinspection PyPep8Naming
    def unpack(self, E):
        """

        :param E:
        :return:
        """
        output = torch.zeros((self.BATCH_N * self.MAX_N * self.MAX_N, E.shape[-1]),
                             device=E.device)
        mask = (self.mask > 0).reshape(-1)
        output[mask] = E
        return output.reshape(self.BATCH_N, self.MAX_N, self.MAX_N, E.shape[-1])


class MLP(nn.Module):
    """
    Stuff.
    """

    def __init__(self, layers, activate_final=True):
        super().__init__()
        self.layers = layers
        nets = []
        for i in range(len(layers) - 1):

            nets.append(nn.Linear(layers[i], layers[i + 1]))
            if i < (len(layers) - 2) or activate_final:
                nets.append(nn.ReLU())

        self.net = nn.Sequential(*nets)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.net(x)


class MLPModel(nn.Module):
    """
    Stuff.
    """

    def __init__(self, internal_d, layer_n,
                 input_d=None, output_d=None,
                 norm='batch'):
        super().__init__()

        # noinspection PyPep8Naming
        LAYER_SIZES = [internal_d] * (layer_n + 1)

        if input_d is not None:
            LAYER_SIZES[0] = input_d
        if output_d is not None:
            LAYER_SIZES[-1] = output_d

        # noinspection PyArgumentEqualDefault
        self.mlp = MLP(LAYER_SIZES, activate_final=True)

        self.normkind = norm
        if self.normkind == 'batch':
            self.norm = nn.BatchNorm1d(LAYER_SIZES[-1])
        elif self.normkind == 'layer':
            self.norm = nn.LayerNorm(LAYER_SIZES[-1])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        y = self.mlp(x)
        if self.normkind == 'batch':
            # noinspection PyPep8Naming,PyShadowingNames
            F = y.shape[-1]

            y_batch_flat = y.reshape(-1, F)
            z_flat = self.norm(y_batch_flat)
            z = z_flat.reshape(y.shape)
            return z
        elif self.normkind == 'layer':
            return self.norm(y)
        else:
            return y


class EVMP(nn.Module):
    """
    Stuff.
    """

    # noinspection PyPep8Naming,PyShadowingNames
    def __init__(self, MAX_N, F, STEP_N, agg_edge='max'):
        super(EVMP, self).__init__()

        self.MAX_N = MAX_N
        self.F = F

        self.e_cell = nn.GRUCell(F, F)
        self.v_cell = nn.GRUCell(F, F)

        self.STEP_N = STEP_N

        self.agg_edge = agg_edge

    def forward(self, e, v, e_mask, v_mask):
        """

        :param e:
        :param v:
        :param e_mask:
        :param v_mask:
        :return:
        """
        # noinspection PyPep8Naming
        BATCH_N = e.shape[0]

        vp = Vpack(BATCH_N, self.MAX_N, self.F, v_mask)
        ep = Epack(BATCH_N, self.MAX_N, self.F, e_mask)

        v_in_f = vp.pack(v)
        e_in_f = ep.pack(e)

        v_h = v_in_f
        e_h = e_in_f

        e_v = torch.zeros_like(v_h)

        for i in range(self.STEP_N):
            v_h = self.v_cell(e_v, v_h)
            v_h_up = vp.unpack(v_h)
            v_e = v_h_up.unsqueeze(1) * v_h_up.unsqueeze(2)
            v_e_p = ep.pack(v_e)
            e_h = self.e_cell(v_e_p, e_h)
            e_h_up = ep.unpack(e_h)
            e_v_up = goodmax(e_h_up, dim=1)

            e_v = vp.pack(e_v_up)

        # noinspection PyUnboundLocalVariable
        return ep.unpack(e_h), v_h_up


class MPMLPOutNet(nn.Module):
    """
    Stuff.
    """

    # noinspection PyPep8Naming,PyDefaultArgument
    def __init__(self, vert_f_in, edge_f_in, MAX_N,
                 layer_n, internal_d_vert,
                 init_noise=0.0,
                 force_lin_init=False, dim_out=4,
                 final_d_out=1024,
                 final_layer_n=1,
                 force_bias_zero=True,
                 edge_mat_norm=False,
                 force_edge_zero=False,
                 mpconfig={},
                 logsoftmax_out=False,
                 chan_out=1,
                 vert_mask_use=True,
                 edge_mask_use=True,
                 pos_out=False,
                 vert_bn_use=True,
                 edge_bn_use=True,
                 mask_val_edges=False,
                 combine_graph_in=False,
                 log_invalid_offset=-1e4,
                 e_bn_pre_mlp=False

                 ):
        """
        MPNet but with a MP output
        """

        super(MPMLPOutNet, self).__init__()

        self.MAX_N = MAX_N  # Maximum number of atoms in dataset?? or features?
        self.vert_f_in = vert_f_in  # Number of vertex features
        self.edge_f_in = edge_f_in  # Number of edge features

        self.dim_out = dim_out  # Set to 4
        self.internal_d_vert = internal_d_vert  # Set to 512 in yml (hidden size for vertices)
        self.layer_n = layer_n  # Set to 8 in yml (number of layers)
        self.edge_mat_norm = edge_mat_norm  # Set to True in yml
        self.force_edge_zero = force_edge_zero  # Set to False in yml

        self.input_v_bn = MaskedBatchNorm1d(vert_f_in)
        self.input_e_bn = MaskedBatchNorm1d(edge_f_in)

        self.mp = create_mp(MAX_N, internal_d_vert,
                            layer_n, mpconfig)  # mpconfig in yml has a name attribute given as "EVMP"

        self.combine_graph_in = combine_graph_in  # Set to False in yml
        extra_mlp_dim = 0
        if self.combine_graph_in:
            extra_mlp_dim = 4
        self.final_mlp_out = MLPModel(final_d_out,
                                      final_layer_n,
                                      input_d=internal_d_vert + extra_mlp_dim)

        self.e_bn_pre_mlp = e_bn_pre_mlp  # Set to True in yml
        if e_bn_pre_mlp:
            self.pre_mlp_bn = MaskedBatchNorm1d(internal_d_vert + extra_mlp_dim)
        self.final_out = nn.Linear(final_d_out,
                                   dim_out)  # final_d_out set to 1024 in yml

        self.chan_out = chan_out  # Set to 1 in yml

        self.init_noise = init_noise  # Set to 0.01

        if force_lin_init:  # Set to True in yml
            self.force_init(init_noise, force_bias_zero)

        # noinspection PyArgumentList
        self.triu_idx = torch.Tensor(triu_indices_flat(MAX_N, k=1)).long()

        # self.softmax_out = softmax_out
        self.logsoftmax_out = logsoftmax_out  # Set to False in yml
        self.pos_out = pos_out  # Set to False in yml

        self.vert_mask_use = vert_mask_use  # Set to True in yml
        self.edge_mask_use = edge_mask_use  # Set to True in yml

        self.vert_bn_use = vert_bn_use  # Set to True in yml
        self.edge_bn_use = edge_bn_use  # Set to True in yml
        self.mask_val_edges = mask_val_edges  # Set to True in yml
        self.log_invalid_offset = log_invalid_offset  # Set to -10 in yml

    def force_init(self, init_noise=None, force_bias_zero=True):
        """

        :param init_noise:
        :param force_bias_zero:
        :return:
        """
        if init_noise is None:
            init_noise = self.init_noise
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_noise < 1e-12:
                    nn.init.xavier_uniform_(m.weight)
                else:

                    nn.init.normal_(m.weight, 0, init_noise)
                if force_bias_zero:
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, v_in, e_in, graph_conn_in,
                vert_mask, possible_val):
        """
        output is:
        [BATCH_N, FLATTEN_LENGHED_N, LABEL_LEVELS, M]
        """

        # noinspection PyPep8Naming
        BATCH_N = v_in.shape[0]

        v_mask = vert_mask.unsqueeze(-1)
        e_mask = (vert_mask.unsqueeze(-1) * vert_mask.unsqueeze(-2)).unsqueeze(-1)

        def last_bn(layer, x, mask):
            """

            :param layer:
            :param x:
            :param mask:
            :return:
            """
            init_shape = x.shape

            x_flat = x.reshape(-1, init_shape[-1])
            mask_flat = mask.reshape(-1)
            x_bn = layer(x_flat, mask_flat)
            return x_bn.reshape(init_shape)

        if self.force_edge_zero:
            e_in[:] = 0

        if self.edge_mat_norm:
            e_in = batch_mat_chan_norm(e_in)

        v_in_bn = last_bn(self.input_v_bn, v_in, v_mask)
        e_in_bn = last_bn(self.input_e_bn, e_in, e_mask)
        #
        # # noinspection PyTypeChecker
        # v = F.pad(v_in_bn, (0, self.internal_d_vert - v_in_bn.shape[-1]), "constant", 0)
        #
        # if self.vert_mask_use:
        #     v = v * v_mask
        #
        # # noinspection PyTypeChecker
        # e = F.pad(e_in_bn,
        #           (0, self.internal_d_vert - e_in_bn.shape[-1]),
        #           "constant", 0)
        # if self.edge_mask_use:
        #     e = e * e_mask
        #
        # e_new, v = self.mp(e, v, e_mask, v_mask)
        #
        # if self.combine_graph_in:
        #     e_new = torch.cat([e_new, graph_conn_in], -1)
        # if self.e_bn_pre_mlp:
        #     e_new = last_bn(self.pre_mlp_bn, e_new, e_mask)
        #
        # e_est = self.final_out(self.final_mlp_out(e_new))
        #
        # e_est = e_est.unsqueeze(-1)
        #
        # assert e_est.shape[-1] == self.chan_out
        #
        # if self.mask_val_edges:
        #     e_est = e_est + (1.0 - possible_val.unsqueeze(-1)) * self.log_invalid_offset
        #
        # a_flat = e_est.reshape(BATCH_N, -1, self.dim_out, self.chan_out)
        # a_triu_flat = a_flat[:, self.triu_idx, :, :]
        #
        # if self.logsoftmax_out:
        #     a_triu_flatter = a_triu_flat.reshape(BATCH_N, -1, 1)
        #     if self.logsoftmax_out:
        #         a_nonlin = F.log_softmax(a_triu_flatter, dim=1)
        #     elif self.softmax_out:
        #         a_nonlin = F.softmax(a_triu_flatter, dim=1)
        #     else:
        #         raise ValueError()
        #
        #     a_nonlin = a_nonlin.reshape(BATCH_N, -1, self.dim_out, 1)
        # else:
        #
        #     a_nonlin = a_triu_flat
        #
        # if self.pos_out:
        #     a_nonlin = F.relu(a_nonlin)
        #
        # return a_nonlin
