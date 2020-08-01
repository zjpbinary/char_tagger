import torch
class Metric(object):
    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class AccuracyMethod(Metric):

    def __init__(self, eps=1e-5):
        super(AccuracyMethod, self).__init__()

        self.tp = 0.0
        self.total = 0.0
        self.eps = eps

    def __call__(self, preds, golds, mask):
        golds = golds.transpose(0, 1)
        lens = mask.sum(dim=0)
        for i, pred in enumerate(preds):
            self.tp += torch.sum(pred == golds[i, :lens[i]]).item()
            self.total += len(pred)

    def __repr__(self):
        return f"Accuracy: {self.accuracy:.2%}"

    def clear(self):
        self.total = 0.0
        self.correct = 0.0

    @property
    def score(self):
        return self.accuracy

    @property
    def accuracy(self):
        return self.tp / (self.total + self.eps)
'''
x = [torch.LongTensor([1,2,3]),torch.LongTensor([1,2,3])]
y = [torch.LongTensor([1,2,1]),torch.LongTensor([1,2,3])]
m = AccuracyMethod()
m(x,y)
print(m.score)
m.clear()
'''
