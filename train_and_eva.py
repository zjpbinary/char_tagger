from metric import *
import torch.nn as nn
m = AccuracyMethod()
def eva_model(model, test_iter, pad_index = 1, device='cpu'):
    model.to(device)
    model.eval()
    for batch in test_iter:
        words = batch.word.to(device)
        ptbtag = batch.ptbtag.to(device)
        chars = batch.char.to(device)
        mask = words.ne(pad_index)
        emit = model.forward(words, chars)
        preds = model.crf.viterbi(emit, mask)
        m(preds, ptbtag, mask)
    print(f'测试集上的精度为:\n {m.score}')
    m.clear()
def train_model(model, train_iter, optimizer, epoch, pad_index=1, device='cpu'):
    model.to(device)
    for e in range(epoch):
        model.train()
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            words = batch.word.to(device)
            ptbtag = batch.ptbtag.to(device)
            chars = batch.char.to(device)
            mask = words.ne(pad_index)
            emit = model.forward(words, chars)
            loss = model.crf.forward(emit, ptbtag, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            print(f'第{e}次迭代，第{i}个batch，loss为{loss}')


