import torch
import torch.nn as nn

class CRF(nn.Module):
    def __init__(self, n_tags):
        super(CRF, self).__init__()
        self.n_tags = n_tags
        self.trans = nn.Parameter(torch.Tensor(n_tags, n_tags))
        # from->to
        self.strans = nn.Parameter(torch.Tensor(n_tags))
        self.etrans = nn.Parameter(torch.Tensor(n_tags))

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.zeros_(self.trans)
        nn.init.zeros_(self.strans)
        nn.init.zeros_(self.etrans)
    def get_logZ(self, emit, mask):
        max_len, batch_size, n_tags = emit.shape
        alpha = self.strans + emit[0]
        # shape: batch_size x n_tags
        for i in range(1, max_len):
            trans_and_emit = self.trans + emit[i].unsqueeze(1)
            scores = trans_and_emit + alpha.unsqueeze(2)
            scores = torch.logsumexp(scores, dim=1)
            # 对每一行求logsumexp, shape: batch_size x n_tags
            mask_i = mask[i].unsqueeze(1).expand_as(alpha)
            # shape: batch_size x n_tags
            alpha[mask_i] = scores[mask_i]
            # 利用mask只更新未结束的句子
        logZ = torch.logsumexp(alpha + self.etrans, dim=1).sum()
        # 对每一列求logsumexp, 再把所有句子的logsumexp相加
        return logZ / batch_size # 返回均值
    def get_score(self, emit, target, mask):
        max_len, batch_size, n_tags = emit.shape
        scores = emit.new_zeros(max_len, batch_size)
        # 加上转移得分
        scores[1:] += self.trans[target[:-1], target[1:]]
        # 加上emit得分，收集目标类别的得分
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # 利用掩码提取所有句子得分,并求和
        score = scores.masked_select(mask).sum() # shape: 1
        # 加入所有句子的起始转移得分
        score += self.strans[target[0]].sum()
        # 加入所有句子的结束转移得分
        ends = mask.sum(dim=0).view(1,-1) - 1
        # shape: 1 x batch_size 记录每句话的结束位置
        score += self.etrans[target.gather(dim=0, index=ends)].sum()
        return score / batch_size
    def forward(self, emit, target, mask):
        logZ = self.get_logZ(emit, mask)
        score = self.get_score(emit, target, mask)
        return logZ - score
    def viterbi(self, emit, mask):
        max_len, batch_size, n_tags = emit.shape
        lens = mask.sum(0)
        # 记录每句的实际长度
        delta = emit.new_zeros(max_len, batch_size, n_tags)
        paths = emit.new_zeros(max_len, batch_size, n_tags, dtype=torch.long)
        delta[0] = self.strans + emit[0]
        for i in range(1, max_len):
            trans_and_emit = self.trans + emit[i].unsqueeze(1)
            # 广播 n_tags x n_tags + batch_size x 1 x n_tags
            # 等于 batch_size x n_tags x n_tags
            scores = trans_and_emit + delta[i-1].unsqueeze(2)
            # batch_size x n_tags x n_tags
            delta[i], paths[i] = torch.max(scores, dim = 1)


        preds = []
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length-1, i] + self.etrans)
            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            preds.append(paths.new_tensor(predict).flip(0))
            #flip反转后加入列表
        return preds
'''
crf = CRF(10)
target = torch.LongTensor([[2,3],[4,5],[1,2]])
x = torch.randn(3,2,10)
mask = torch.BoolTensor([[1,1],[1,1],[1,0]])
print(crf.get_score(x, target, mask))
'''
