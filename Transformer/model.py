# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 下午4:12
 @FileName: model.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadBlock(nn.Module):
    def __init__(self, n_input, n_head=6):
        super().__init__()
        self.combined_projection = nn.Linear(n_input, 2 * (n_input // n_head) * n_head + (n_input // 2) * n_head)
        self.output_projection = nn.Linear((n_input // 2) * n_head, n_input)

        nn.init.xavier_normal_(self.combined_projection.weight, gain=0.1)
        nn.init.xavier_normal_(self.output_projection.weight, gain=0.1)

        self._scale = (n_input // n_head) ** 0.5
        self.att_dim = (n_input // n_head) * n_head
        self.num_heads = n_head

    def forward(self, representations):
        batch_size, timesteps, _ = representations.size()
        combined_projection = F.leaky_relu(self.combined_projection(representations), inplace=True)
        queries, keys, *values = combined_projection.split(self.att_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        values_per_head = values.view(batch_size, timesteps, self.num_heads, values.size(-1) // self.num_heads)
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * self.num_heads, timesteps,
                                               values.size(-1) // self.num_heads)

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * self.num_heads, timesteps,
                                                 self.att_dim // self.num_heads)

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * self.num_heads, timesteps, self.att_dim // self.num_heads)

        similarities = queries_per_head.bmm(keys_per_head.transpose(2, 1)) / self._scale

        similarities = F.softmax(similarities, 2)

        outputs = similarities.bmm(values_per_head)

        outputs = outputs.view(batch_size, self.num_heads, timesteps, values.size(-1) // self.num_heads)
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, values.size(-1))

        return representations + F.leaky_relu(self.output_projection(outputs), inplace=True)


class SelfAttention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6):
        super().__init__()
        self.n_head = n_head
        self.att = nn.ModuleList()
        for _ in range(n_layer):
            en = MultiHeadBlock(n_hidden)
            ln = nn.LayerNorm(n_hidden)
            self.att.append(nn.Sequential(en, ln))

    def forward(self, representations):
        for one in self.att:
            representations = one(representations)
        return representations


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 4, embedding_dim=n_embedding - (n_embedding // 4))
        self.pos_size = 512
        self.pos_embedding = nn.Embedding(self.pos_size, n_embedding // 4)
        self.n_embedding = n_embedding - n_embedding // 4
        self.attention = SelfAttention(n_embedding, n_layer)
        self.projection = nn.Linear(n_embedding, n_embedding - (n_embedding // 4))
        self.trans = nn.Linear(n_embedding - (n_embedding // 4), vocab_size + 4, bias=False)
        self.embedding.weight = self.trans.weight

    def forward(self, inputs):
        [seq, index, target] = inputs
        word_embedding = self.embedding(seq)
        pos = torch.arange(seq.size(1)).cuda()
        pos %= self.pos_size
        pos = pos.expand_as(seq)
        pos_embedding = self.pos_embedding(pos)
        encoder_representations = torch.cat([word_embedding, pos_embedding], -1)
        encoder_representations = self.attention(encoder_representations)
        encoder_representations = F.leaky_relu(self.projection(encoder_representations), True)
        hidden = encoder_representations.gather(1, index.unsqueeze(2).expand(index.size(0), index.size(1),
                                                                             self.n_embedding))
        mask_loss = F.cross_entropy(F.log_softmax(self.trans(hidden.contiguous().view(-1, self.n_embedding)), 1),
                                    target.contiguous().view(-1))
        return mask_loss

    def inference(self, seq):
        embedding = self.embedding(seq)
        encoder_representations, _ = self.encoder(embedding)
        encoder_representations = F.leaky_relu(self.projection(encoder_representations), inplace=True)
        encoder_representations = self.attention(encoder_representations)
        encoder_representations = F.leaky_relu(self.projection(encoder_representations), True)
        return encoder_representations
