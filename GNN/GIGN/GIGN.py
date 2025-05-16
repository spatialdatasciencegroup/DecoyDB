# %%
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from HIL import HIL


class GIGN(nn.Module):
    def __init__(self, node_dim, hidden_dim,layer_num=3):
        super().__init__()
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        print(self.lin_node[0].weight.shape)
        self.gconv1 = HIL(hidden_dim, hidden_dim)
        self.gconv2 = HIL(hidden_dim, hidden_dim)
        self.gconv3 = HIL(hidden_dim, hidden_dim)
        self.gconv = nn.ModuleList()
        self.layer_num = layer_num
        for i in range(layer_num):
            self.gconv.append(HIL(hidden_dim, hidden_dim))
        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x, edge_index_intra, edge_index_inter, pos = \
        data.x, data.edge_index_intra, data.edge_index_inter, data.pos
        label = data.y
        # print(f"x shape: {x.shape}, weight shape: {self.lin_node[0].weight.shape}")
        x = self.lin_node(x)
        # x = self.gconv1(x, edge_index_intra, edge_index_inter, pos)
        # x = self.gconv2(x, edge_index_intra, edge_index_inter, pos)
        # x = self.gconv3(x, edge_index_intra, edge_index_inter, pos)
        x = self.gconv[0](x, edge_index_intra, edge_index_inter, pos)
        for i in range(self.layer_num-1):
            x = x+self.gconv[i+1](x, edge_index_intra, edge_index_inter, pos)
        x1 = global_add_pool(x, data.batch)
        x,h = self.fc(x1)
        
        return x.view(-1),label,h

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            # if j == self.n_FC_layer - 1:
            #     self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
        self.output = nn.Linear(self.d_FC_layer, n_tasks)
    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        x = self.output(h)
        return x,h

# %%