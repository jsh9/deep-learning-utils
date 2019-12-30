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
twenty_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42
)

twenty_test = fetch_20newsgroups(
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
    batch_size=256,
)

tokenizer_test = dlu.data_utils.WordTokenizer(existing_vocab=vocab, max_length=ml)
test_data, _ = dlu.data_utils.create_text_data_pack(
    texts_test,
    labels_test,
    tokenizer_test,
)

#%%--------------- Initialize text CNN model parameters -----------------------
WORD_VECTOR_DIM = 100
embed_size = WORD_VECTOR_DIM
kernel_sizes = [3, 4, 5]
num_channels = [100, 100, 100]

model = dlu.textCNN.TextCNN(
    vocab=vocab,
    embed_size=embed_size,
    kernel_sizes=kernel_sizes,
    num_channels=num_channels,
    num_classes=len(categories)
)

glove_wordvec = torchtext.vocab.GloVe(name='6B', dim=WORD_VECTOR_DIM, cache='./glove')
model.embedding.weight.data.copy_(
    dlu.textCNN.load_pretrained_embedding(vocab.itos, glove_wordvec))
model.constant_embedding.weight.data.copy_(
    dlu.textCNN.load_pretrained_embedding(vocab.itos, glove_wordvec))
model.constant_embedding.weight.requires_grad = False

#%%----------------- Training -------------------------------------------------
lr, num_epochs = 0.005, 15
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr
)
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dlu.train_test_utils.train(
    train_iter=train_iter,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=num_epochs,
    unpack_repack_fn=dlu.textCNN.unpack_data_for_textCNN,
    test_data=test_data,
    device=device,
    eval_test_accuracy=True,
    eval_test_AUC=False
)

#%%--------------- Evaluate after training ------------------------------------
model.to('cpu')
X_test = test_data['padded_token_IDs'].to("cpu")
y_true_test = test_data['labels'].detach().numpy()
y_pred_test_ = model(X_test)
y_pred_class_test = torch.argmax(y_pred_test_, dim=1).detach().numpy()

from sklearn.metrics import accuracy_score

print('Accuracy = %.3f' % accuracy_score(y_true_test, y_pred_class_test))

