#  -*- coding: UTF-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

device = torch.device("cuda")


class InputEmbedding(nn.Module):
    def __init__(self, input_shape, d_model=4):
        super().__init__()
        self.input_shape = input_shape
        self.d_model = d_model
        self.input_embedding = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_shape[2], self.d_model)),
            ('activation', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(self.d_model, self.d_model))
        ]))

    def forward(self, x):
        x = self.input_embedding(x)
        return x.transpose(2, 3)


class ScaleDotProductAttention_spatial(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # print(q.size())
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # print('attn:',attn.size())
        x = torch.matmul(attn, v)
        return x, attn


class MultiHeadAttention_spatial(nn.Module):  # d_model*seq_len = n_head * d_k
    def __init__(self, d_model, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.w_qs = nn.Linear(d_model * seq_len, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model * seq_len, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model * seq_len, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model * seq_len, bias=False)
        self.attention = ScaleDotProductAttention_spatial(temperature=d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model * seq_len, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # print('d_model', self.d_model)
        # print('seq_len', self.seq_len)
        # print("q:",q.size())
        # print("k:",k.size())
        # print("v:",v.size())
        sz_b, n_q, n_k, n_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, n_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, n_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, n_v, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, n_q, -1)
        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        # print("q:",q.size())
        # print("attn:",attn.size())
        return q, attn


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  ## position-wise, bias=True
        self.w_2 = nn.Linear(d_hid, d_in)  ##
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class Transformer_spatial(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.attn = MultiHeadAttention_spatial(self.d_model, self.n_head, self.d_k, self.d_v, self.seq_len,
                                               dropout=dropout)
        self.ff = FeedForward(self.d_model * self.seq_len, self.d_inner, dropout=dropout)

    def forward(self, x, v, mask=None):
        batch_size, seq_len, features, cells = x.shape
        x = x.reshape(batch_size, seq_len * features, cells).transpose(1, 2)
        v = v.reshape(batch_size, seq_len * features, cells).transpose(1, 2)
        x, attn = self.attn(x, x, v, mask=mask)
        x = self.ff(x)
        x = x.transpose(1, 2).view(batch_size, seq_len, features, cells)
        return x, attn


class ScaleDotProductAttention_temporal(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(3, 4))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        x = torch.matmul(attn, v)
        return x, attn


class MultiHeadAttention_temporal(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaleDotProductAttention_temporal(temperature=d_k ** 0.5, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        sz_b, n_q, n_k, n_v, len_q, len_k, len_v = \
            q.size(0), q.size(1), k.size(1), v.size(1), q.size(2), k.size(2), v.size(2)
        residual = q

        q = self.w_qs(q).view(sz_b, n_q, len_q, self.n_head,
                              self.d_k)
        k = self.w_ks(k).view(sz_b, n_k, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, n_v, len_v, self.n_head, self.d_v)
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(2, 3).contiguous().view(sz_b, n_q, len_q, -1)
        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q, attn


class Transformer_temporal(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.attn = MultiHeadAttention_temporal(self.d_model, self.n_head, self.d_k, self.d_v, self.seq_len,
                                                dropout=dropout)
        self.ff = FeedForward(self.d_model, self.d_inner, dropout=dropout)

    def forward(self, x, v, mask=None):
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        x, attn = self.attn(x, x, v, mask=mask)
        x = self.ff(x)
        x = x.permute(0, 2, 3, 1)
        return x, attn


class Transformer_st(nn.Module):
    def __init__(self, n_layers, d_model, d_inner, nheads_spatial, nheads_temporal, dk_s, dv_s, dk_t, dv_t, seq_len,
                 dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.nheads_spatial = nheads_spatial
        self.nheads_temporal = nheads_temporal
        self.dk_s = dk_s
        self.dv_s = dv_s
        self.dk_t = dk_t
        self.dv_t = dv_t
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.spatial_layers = nn.ModuleList([
            Transformer_spatial(d_model, d_inner, nheads_spatial, dk_s, dv_s, seq_len, dropout=dropout)
            for _ in range(self.n_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            Transformer_temporal(d_model, d_inner, nheads_temporal, dk_t, dv_t, seq_len, dropout=dropout)
            for _ in range(self.n_layers)
        ])

    def forward(self, x, mask=None, return_attns=True):
        attn_spatial_list = []
        attn_temporal_list = []

        enc_o = x
        for i in range(self.n_layers):
            x = enc_o
            enc_o, attn_s = self.spatial_layers[i](x, enc_o, mask)
            enc_o = enc_o + x
            enc_o, attn_t = self.temporal_layers[i](x, enc_o, mask)
            enc_o = enc_o + x
            attn_spatial_list.append(attn_s)
            attn_temporal_list.append(attn_t)
        if return_attns:
            return enc_o, attn_spatial_list, attn_temporal_list
        return enc_o


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        # print("input.shape:",input.contiguous().shape)
        new_features = super(_DenseLayer, self).forward(input.contiguous())
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([input, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class iLayer(nn.Module):
    def __init__(self):
        super(iLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(1))

    def forward(self, x):
        w = self.w.expand_as(x)
        return x * w


class Transformer(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, growth_rate=12, num_init_features=12, bn_size=4,
                 drop_rate=0.1, nb_flows=1, fusion=1, maps=1, d_model=12, d_inner=128, nheads_spatial=6,
                 nheads_temporal=6, layers=3, dk_s=16, dk_t=16,
                 flags_meta=0, flags_cross=0):
        super().__init__()
        self.input_shape = input_shape
        self.meta_shape = meta_shape
        self.cross_shape = cross_shape
        self.filters = num_init_features
        self.channels = nb_flows
        self.fusion = fusion
        self.maps = maps
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters,) + self.input_shape[-2:]
        self.seq_len = self.input_shape[1]
        self.d_model = d_model
        self.d_inner = d_inner
        self.nheads_spatial = nheads_spatial
        self.nheads_temporal = nheads_temporal
        self.layers = layers
        self.dk_s = dk_s
        self.dv_s = dk_s
        self.dk_t = dk_t
        self.dv_t = dk_t

        self.inp_emb = InputEmbedding(self.input_shape, self.d_model)
        # self.dropout = nn.Dropout(drop_rate)
        # self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.transformer = nn.Sequential()
        transformer = Transformer_st(self.layers, self.d_model, self.d_inner, self.nheads_spatial, self.nheads_temporal,
                                     self.dk_s, self.dv_s,
                                     self.dk_t, self.dv_t, self.seq_len, drop_rate, )
        self.transformer.add_module('transformer', transformer)
        self.trg_word_prj = nn.Linear(self.input_shape[1], 1, bias=False)

        # Each denseblock
        self.features = nn.Sequential()

        if (flags_meta == 1) & (flags_cross == 1):
            num_features = self.d_model * 3
        elif (flags_meta == 1) & (flags_cross == 0):
            num_features = self.d_model * 2
        elif (flags_meta == 0) & (flags_cross == 1):
            num_features = self.d_model * 2
        elif (flags_meta == 0) & (flags_cross == 0):
            num_features = self.d_model
        # block_config = [6, 6, 6]
        block_config = [2, 2, 2]
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)  # theta=0.5
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # print(num_features)

        # change from batch norm to relu
        self.features.add_module('relulast', nn.ReLU(inplace=True))

        self.features.add_module('convlast', nn.Conv2d(num_features, nb_flows,
                                                       kernel_size=1, padding=0, bias=False))

        self.tanh = nn.Tanh()

        sqlen = self.input_shape[1] * self.input_shape[2]
        row = self.h
        col = self.w
        out_feature = num_features * 6 if num_features * 6 < sqlen * row * col else num_features
        self.mnorm1 = nn.BatchNorm2d(nb_flows)
        self.mconv1 = nn.Conv2d(nb_flows, out_feature, kernel_size=3, stride=1, padding=1, bias=False)
        self.mconv2 = nn.Conv2d(out_feature, sqlen * row * col, kernel_size=3, stride=1, padding=1, bias=False)
        self.mconv3 = nn.Conv2d(sqlen * row * col, 1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, out, meta=None, cross=None):
        batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
        x = out
        # print('out.size:',out.size())
        out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3)
        # print("before input_embbeding:", out.size())
        out = self.inp_emb(out)
        # print("after position encoding and before transformer:", out.size())
        out, attn_spatial_list, attn_temporal_list = self.transformer(out)
        # print("after transformer:", out.size())
        out = self.trg_word_prj(out.permute(0, 2, 3, 1)).squeeze(-1)
        # print("after dimensional change:", out.size())
        out = out.view(batch_size, -1, row_of_cells, col_of_cells)
        # print("before densenet:", out.size())
        out = self.features(out)

        out = self.mnorm1(out)
        out = self.mconv1(out)
        out = self.mconv2(out)
        # print("origin_x.size:,", x.shape)
        x = x.view(batch_size, 1, -1)
        # print("x.size:,", x.shape)
        x = torch.cat((x,) * row_of_cells * col_of_cells, dim=1)
        # print("cat.size:,", x.shape)
        out = self.tanh(out)
        weight = torch.reshape(out, (
            batch_size, row_of_cells * col_of_cells, row_of_cells * col_of_cells * seq_len * features))
        # print("out.size:,", out.shape)
        # print("weight.size:,", weight.shape)
        out = torch.mul(x, weight)
        out = torch.transpose(out, 1, 2)
        out = torch.reshape(out,
                            (batch_size, row_of_cells * col_of_cells * seq_len * features, row_of_cells, col_of_cells))
        # print("out.size:", out.shape)
        out = self.mconv3(out)

        out = torch.sigmoid(out)
        # # print("result:", out.size())
        # # print("out:", out[0])
        return out, None
        # return out, weight  # , attn_spatial_list, attn_temporal_list
