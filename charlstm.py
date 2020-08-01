import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from config import *

class CharLSTM(nn.Module):
    def __init__(self, n_char, n_embed=args.char_embed, n_out=args.char_out):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(n_char, n_embed)

        self.lstm = nn.LSTM(input_size=n_embed, hidden_size=n_out, bidirectional=True)
        self.n_out = n_out

    def forward(self, x_char):
        # x_char shape:batch_size x max_len x word_len
        # embedding shape:batch_size x max_len x word_len x n_embed
        mask = x_char.ne(1)
        lens = mask.sum(-1)
        #lens记录了 每个单词的实际长度
        char_mask = lens.gt(0)
        #删除实际长度为0的词
        embedding = self.embed(x_char[char_mask])
        #embeddding shape:有效词数 x 固定词长度 x n_embed
        pad_seq = pack_padded_sequence(embedding, lens[char_mask], True, False)
        out, (h_n, _) = self.lstm(pad_seq)
        '''
        out为packsquence对象
        h_n保存了每一层，最后一个time step的输出h，如果是双向LSTM，单独保存前向和后向的最后一个time step的输出h。
        此处h_n shape: 2 x 有效词数 x n_out
        c_n与h_n一致，只是它保存的是c的值。
        '''
        h_n = torch.cat(torch.unbind(h_n), -1) #h_n shape: 有效词数 x (2*n_out)
        embed = h_n.new_zeros(*lens.shape, self.n_out*2)
        # embed shape: batch_size x max_len x n_out
        embed = embed.masked_scatter_(char_mask.unsqueeze(-1), h_n)
        #arget.masked_scatter_(mask, source) source是表示用来替换（改变）的值，而mask只有0,1；0表示不替换目标值（或者叫不改变）；1是表示要替换目标值。
        return embed
        # return shape:  batch_size x max_len x (n_out*2)

