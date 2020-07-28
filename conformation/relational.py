""" Relational network definition """
import numpy as np
import torch
import torch.nn.functional as F

from conformation.utils import to_undirected


class RelationalNetwork(torch.nn.Module):
    """ Relational network definition """

    def __init__(self, hidden_size=256, num_layers=32, num_edge_features=None, num_vertex_features=None,
                 final_linear_size=1024, final_output_size=1, cnf=False):
        super(RelationalNetwork, self).__init__()
        self.hidden_size = hidden_size  # Internal feature size
        self.num_layers = num_layers  # Number of relational layers
        self.num_edge_features = num_edge_features  # Number of input edge features
        self.num_vertex_features = num_vertex_features  # Number of input vertex features
        self.final_linear_size = final_linear_size  # Number of nodes in final linear layer
        self.final_output_size = final_output_size  # Number of nodes in final output layer
        self.edge_featurize = torch.nn.Linear(self.num_edge_features,
                                              self.hidden_size)  # Initial linear layer for featurization of edge feat.
        self.vertex_featurize = torch.nn.Linear(self.num_vertex_features,
                                                self.hidden_size)  # Initial layer for featurization of vertex features
        self.L_e = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size) for _ in
                                        range(self.num_layers)])  # Linear layers for edges
        self.L_v = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size) for _ in
                                        range(self.num_layers)])  # Linear layers for vertices
        self.edge_batch_norm = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(self.hidden_size) for _ in range(self.num_layers)])  # Batch norms for edges (\phi_e)
        self.vertex_batch_norm = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_size) for _ in
                                                      range(self.num_layers)])  # Batch norms for vertices (\phi_v)
        self.gru = torch.nn.ModuleList(
            [torch.nn.GRU(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])  # GRU cells
        self.final_linear_layer = torch.nn.Linear(self.hidden_size, self.final_linear_size)  # Final linear layer
        self.output_layer = torch.nn.Linear(self.final_linear_size, self.final_output_size)  # Output layer
        self.cnf = cnf  # Whether or not we are using this for a conditional normalizing flow

    def forward(self, batch):
        """
        Forward pass.
        :param batch: Data batch.
        :return:
        """
        e_ij_in = self.edge_featurize(batch.edge_attr)  # Featurization
        v_i_in = self.vertex_featurize(batch.x)

        for k in range(self.num_layers):
            e_ij = self.L_e[k](e_ij_in)  # Linear layer for edges
            v_i_prime = self.L_v[k](v_i_in)  # Linear layer for vertices
            e_ij_prime = F.relu(self.edge_batch_norm[k](torch.stack(
                [e_ij[edge_num] + v_i_prime[batch.edge_index[0][edge_num]] + v_i_prime[batch.edge_index[1][edge_num]]
                 for edge_num in range(
                    e_ij.size(0))])))  # Add pairwise vertex features to edge features followed by batch norm and ReLU
            undirected_edge_index = to_undirected(batch.edge_index,
                                                  batch.num_nodes)  # Full set of undirected edges for bookkeeping
            # noinspection PyTypeChecker
            v_i_e = torch.stack([torch.max(e_ij_prime[np.array([np.intersect1d(
                np.where(batch.edge_index[0] == min(vertex_num, i)),
                np.where(batch.edge_index[1] == max(vertex_num, i))) for i in np.array(
                undirected_edge_index[1][np.where(undirected_edge_index[0] == vertex_num)])]).flatten()], 0)[0] for
                                 vertex_num in
                                 range(batch.num_nodes)])  # Aggregate edge features
            gru_input = v_i_e.view(1, batch.num_nodes, self.hidden_size)  # Resize GRU input
            gru_hidden = v_i_in.view(1, batch.num_nodes, self.hidden_size)  # Resize GRU hidden
            gru_output, _ = self.gru[k](gru_input, gru_hidden)  # Compute GRU output
            v_i_c = F.relu(self.vertex_batch_norm[k](
                gru_output.view(batch.num_nodes, self.hidden_size)))  # Apply batch norm and ReLU to GRU output
            v_i_in = v_i_c + v_i_in  # Add residual connection to vertex input
            e_ij_in = e_ij_prime + e_ij_in  # Add residual connection to edge input

        e_ij_final = self.final_linear_layer(e_ij_in)  # Compute final linear layer
        preds = self.output_layer(e_ij_final)  # Output layer

        if self.cnf:
            return e_ij_in
        else:
            return preds
