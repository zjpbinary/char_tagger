from loaddata import *
from lstm_crf import *
from train_and_eva import *

device = torch.device(args.gpu_id)
print(device)
model = LSTM_CRF(n_words, n_tags, n_chars, pad_index,
                 embed_dim=args.embed_dim,
                 lstm_dim=args.lstm_dim,
                 hid_dim=args.hid_dim,
                 dropout=args.dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


train_model(model, train_iter, optimizer, args.epoch, device=device)
eva_model(model, test_iter, device=device)