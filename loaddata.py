from pprint import pprint
from torchtext import data
from torchtext import datasets
from config import *

# Now lets try both word and character embeddings
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# We'll use NestedField to tokenize each word into list of chars
CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>")
CHAR = data.NestedField(CHAR_NESTING, init_token="<bos>", eos_token="<eos>")

fields = [(('word', 'char'), (WORD, CHAR)), (None, None), ('ptbtag', PTB_TAG)]
train, val, test = datasets.UDPOS.splits(fields=fields)

#build_vocab
WORD.build_vocab(train.word, val.word, test.word, min_freq=3)
CHAR.build_vocab(train.char, val.char, test.char)
PTB_TAG.build_vocab(train.ptbtag)

n_words = len(WORD.vocab.itos)
n_tags = len(PTB_TAG.vocab.itos)
n_chars = len(CHAR.vocab.itos)

print(f'n_words {n_words}')
print(f'n_tags {n_tags}')
print(f'n_chars {n_chars}')

unk_index = WORD.vocab.unk_index
pad_index = WORD.vocab.stoi['<pad>']

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size, sort_within_batch=True)
#batch.char shape: batch_size x max_len x word_len
#batch.word shape: max_len x batch_size
#batch.ptbtag shape: max_len x batch_size
