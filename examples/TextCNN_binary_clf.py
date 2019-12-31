# -*- coding: utf-8 -*-
"""
Binary classification using text CNN.

Data set: IMDb sentiment, 2-class (SST-2)
"""
import torch
import torchtext
import pandas as pd
import deep_learning_utils as dlu

#%%-------------------- Load IMDB sentiment data ------------------------------
print('Loading data... ', end='')
path_train = './data/SST-2/train.tsv' # 6920 records in total
path_test = './data/SST-2/test.tsv'  # 1821 records in total
df_train = pd.read_csv(path_train, delimiter='\t', header=None)
df_test = pd.read_csv(path_test, delimiter='\t', header=None)
print('done.')

#%%---------------- Prepare data ----------------------------------------------
texts_train = list(df_train[0])
labels_train = list(df_train[1])

texts_test = list(df_test[0])
labels_test = list(df_test[1])

ml = 40  # how many words in each sentence to use for training/testing

tokenizer_train = dlu.data_utils.WordTokenizer(min_freq=5, max_length=ml)
train_iter, vocab = dlu.data_utils.create_text_data_iter(
    texts_train,
    labels_train,
    tokenizer_train,
    batch_size=128,
)

tokenizer_test = dlu.data_utils.WordTokenizer(existing_vocab=vocab, max_length=ml)
test_iter, _ = dlu.data_utils.create_text_data_iter(
    texts_test,
    labels_test,
    tokenizer_test,
    batch_size=128,
)

#%%--------------- Initialize text CNN model parameters -----------------------
WORD_VECTOR_DIM = 100
embedding_dim = WORD_VECTOR_DIM

model = dlu.textCNN.TextCNNClassifier(
    vocab=vocab,
    embedding_dim=embedding_dim,
)

glove_wordvec = torchtext.vocab.GloVe(name='6B', dim=WORD_VECTOR_DIM, cache='./glove')
model.populate_embedding_layers_with_pretrained_word_vectors(glove_wordvec)

#%%----------------- Training -------------------------------------------------
lr, num_epochs = 0.01, 4#10
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr
)
loss_fn = torch.nn.CrossEntropyLoss()
unpack_repack_fn = dlu.textCNN.unpack_data_for_textCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dlu.train_test_utils.train(
    train_iter=train_iter,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=num_epochs,
    unpack_repack_fn=unpack_repack_fn,
    test_iter=test_iter,
    device=device,
    eval_test_accuracy=True,
    eval_test_AUC=True,
    eval_each_batch=False,
)

#%%--------------- Evaluate after training ------------------------------------
test_scores, eval_txt = dlu.train_test_utils.eval_model(
    model=model,
    test_iter=test_iter,
    loss_fn=loss_fn,
    unpack_repack_fn=unpack_repack_fn,
    static_options_to_model=dict(),
    training_device=device,
    eval_on_CPU=False,
    eval_accuracy=True,
    eval_AUC=True,
    eval_R2=False,
    verbose=True
)
