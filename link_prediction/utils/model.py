import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEncode(nn.Module):
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t))
        return output


class TimeSketch(nn.Module):
    def __init__(self, dim, ignore_zero):
        super(TimeSketch, self).__init__()
        self.ignore_zero = ignore_zero
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        
        num_edge = t.shape[0]//3 # [batch_size, topk, 1]
        if self.ignore_zero:      
            index = (t==0).nonzero()

        t=torch.unsqueeze(t,-1)
        x = torch.cos(self.w(t)) # [batch_size, topk, time_dim]

        if self.ignore_zero:
            x[index[:,0],index[:,1]]=0
        x = torch.mean(x,dim=1) # [batch_size, time_dim]
        x= torch.nn.functional.normalize(x,dim=1)
        src = x[:num_edge] # [num_edge,time_dim]
        dst = x[num_edge:2*num_edge]
        neg = x[2*num_edge:]

        pos = torch.sum(src.mul(dst),dim=1).unsqueeze(-1)
        neg = torch.sum(src.mul(neg),dim=1).unsqueeze(-1)
        return pos, neg


class MLPTime(nn.Module):
    def __init__(self, dim):
        super(MLPTime, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim,dim)    
        self.edge_predictor = EdgePredictor_per_node(dim)
        self.combiner = Combiner(10)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
        self.edge_predictor.reset_parameters()
        self.combiner.reset_parameters()

    def forward(self, t):
        with torch.no_grad():
            index = (t==0).nonzero()

            t=torch.unsqueeze(t,-1)
            x = torch.cos(self.w(t)) # [batch_size, topk, time_dim]

            x[index[:,0],index[:,1]]=0
            x = self.layernorm(x)
            x = torch.mean(x, dim=1).squeeze(dim=1)
            x = self.mlp_head(x)
        
        x = self.edge_predictor(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MixerBlock(nn.Module):
    def __init__(self, per_graph_size, dims, 
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 dropout=0, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')


        if 'token' in self.module_spec:
            self.token_layernorm = nn.LayerNorm(dims)
            self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
            
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        

    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.token_layernorm.reset_parameters()
            self.token_forward.reset_parameters()

        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)

        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)

        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, per_graph_size, time_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False,
                 device = 'cpu'
                ):
        super().__init__()

        self.per_graph_size = per_graph_size
        self.num_layers = num_layers
        self.time_encoder = TimeEncode(time_channels)
        self.feat_encoder = nn.Linear(time_channels, time_channels) 
        self.layernorm = nn.LayerNorm(time_channels)
        self.mlp_head = nn.Linear(time_channels,time_channels)
        self.mixer_blocks = torch.nn.ModuleList()
        self.time_channels = time_channels
        self.device = device

        for ell in range(num_layers):
            if module_spec is None:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, time_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer).to(self.device)
                    )
            else:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, time_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer).to(self.device)
                    )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()


    def forward(self, delta_times, inds, batch_size):
        time_encodings = self.time_encoder(delta_times) 
        time_encodings = self.feat_encoder(time_encodings)

        x = torch.zeros((batch_size * self.per_graph_size, self.time_channels)).to(self.device)

        x[inds] = time_encodings # add 0 gaps if node N's num_nei < topk, add to max_edge

        x = torch.split(x, self.per_graph_size) # -> tuples

        x = torch.stack(x) # [batch_size=num(3types nodes), graph_size=max_edge, hidden_dim]

        for i in range(self.num_layers):
            x = self.mixer_blocks[i](x)
            
        x = self.layernorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x


class Combiner(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.linear1 = torch.nn.Linear(2, dim)
        self.linear2 = torch.nn.Linear(dim, 1)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, h):
        num_edge = h.shape[0] // 2
        x = self.linear1(h)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x[:num_edge], x[num_edge:]


class EdgePredictor_per_node(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.src_fc = torch.nn.Linear(dim, 100)
        self.dst_fc = torch.nn.Linear(dim, 100)
        self.out_fc = torch.nn.Linear(100, 1)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h, num_neg=1):
        num_edge = h.shape[0] // (num_neg + 2)
        
        # encode pos embeds
        h_src = self.src_fc(h[:num_edge])

        # encode pos and neg dest embeds
        h_pos_dst = self.dst_fc(h[num_edge:2*num_edge])

        h_neg_dst = self.dst_fc(h[2*num_edge:(num_neg+2)*num_edge]) # [num_neg*num_edge, embed_dim]
        h_src_repeated = h_src.repeat(num_neg, 1) # [num_neg*num_edge, embed_dim]

        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)

        h_neg_edge = torch.nn.functional.relu(h_src_repeated + h_neg_dst)

        # [batch_size, 1]
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
    

class Mixer_per_node(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Mixer_per_node, self).__init__()

        self.dim = edge_predictor_configs['dim']
        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        self.base_model = MLPMixer(**mlp_mixer_configs)
        self.combiner = Combiner(10)
        self.reset_parameters()            

    def reset_parameters(self):
        self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        self.combiner.reset_parameters()

    def forward(self, delta_times, all_inds, batch_size, num_neg=1):  
        x = self.base_model(delta_times, all_inds, batch_size)
        pred_pos, pred_neg = self.edge_predictor(x, num_neg)
        return pred_pos, pred_neg


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=2)
        h = self.act(self.fc1(x))
        return self.fc2(h).mean(dim=0)


class EdgePredictor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.src_fc = torch.nn.Linear(dim, 100)
        self.dst_fc = torch.nn.Linear(dim, 100)
        self.out_fc = torch.nn.Linear(100, 1)
        self.norm = torch.nn.LayerNorm(100)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h):

        num_edge = h.shape[0] // 3
        h_src = self.src_fc(h[:num_edge])
        dst = self.dst_fc(h[num_edge:])
        h_pos_dst = dst[:num_edge]
        h_neg_dst = dst[num_edge:]

        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src + h_neg_dst)

        h_pos_edge = self.norm(h_pos_edge)
        h_neg_edge = self.norm(h_neg_edge)

        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


class NodeClassificationModel(nn.Module):
    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
