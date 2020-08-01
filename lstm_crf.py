from crf import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from charlstm import *
class LSTM_CRF(nn.Module):
    def __init__(self, n_vocabs, n_tags, n_char, pad_index,
                 embed_dim=150, lstm_dim=150, hid_dim=150, dropout=0.5):
        super(LSTM_CRF, self).__init__()
        self.pad_index = pad_index
        self.embedding = nn.Embedding(n_vocabs, embed_dim)
        self.charlstm = CharLSTM(n_char)
        self.lstm = nn.LSTM(450, lstm_dim, bidirectional=True)
        self.hid = nn.Linear(lstm_dim*2, hid_dim, bias=True)
        self.out = nn.Linear(hid_dim, n_tags, bias=True)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(n_tags)
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.hid.weight)
        nn.init.xavier_uniform_(self.out.weight)
    def forward(self, x, x_char):
        # x shape: max_len x batch_size
        mask = x.ne(self.pad_index)
        emb = self.dropout(self.embedding(x))
        char_emb = self.dropout(self.charlstm(x_char).transpose(0,1))
        embed = torch.cat((emb, char_emb), -1)
        pack_seq = pack_padded_sequence(embed, mask.sum(dim=0))
        lstm_out, _ = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        hidden = self.hid(lstm_out)
        hidden = self.activation(hidden)
        return self.out(hidden)
        #out shape: max_len x batch_size x n_tags






