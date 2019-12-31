# -*- coding: utf-8 -*-
"""
Multi-class classification using text CNN.

Data set: 20 news groups dataset

(https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
"""
import torch
import torchtext
import deep_learning_utils as dlu
from sklearn.datasets import fetch_20newsgroups

#%%------------------- Load raw data ------------------------------------------
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(  # 2257 records in total
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42
)

twenty_test = fetch_20newsgroups(  # 1502 records in total
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=42
)

#%%-------------------- Preliminary data processing ---------------------------
texts_train = twenty_train['data']
labels_train = [int(label) for label in twenty_train['target']]

texts_test = twenty_test['data']
labels_test = [int(label) for label in twenty_test['target']]

assert(isinstance(texts_train, list))
assert(all([isinstance(text, str) for text in texts_train]))
assert(isinstance(labels_train, list))
assert(all([isinstance(label, int) for label in labels_train]))

assert(isinstance(texts_test, list))
assert(all([isinstance(text, str) for text in texts_test]))
assert(isinstance(labels_test, list))
assert(all([isinstance(label, int) for label in labels_test]))

#%%
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
    num_classes=len(categories)
)

glove_wordvec = torchtext.vocab.GloVe(name='6B', dim=WORD_VECTOR_DIM, cache='./glove')
model.populate_embedding_layers_with_pretrained_word_vectors(glove_wordvec)

#%%----------------- Training -------------------------------------------------
lr, num_epochs = 0.005, 10
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
    eval_test_AUC=False,
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
    eval_AUC=False,
    eval_R2=False,
    verbose=True
)
