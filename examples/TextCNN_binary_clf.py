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
df_train = pd.read_csv(path_train, delimiter='\t', header=None)[:2000]
df_test = pd.read_csv(path_test, delimiter='\t', header=None)[:400]
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
embedding_dim = WORD_VECTOR_DIM

model = dlu.textCNN.TextCNN(
    vocab=vocab,
    embedding_dim=embedding_dim,
)

glove_wordvec = torchtext.vocab.GloVe(name='6B', dim=WORD_VECTOR_DIM, cache='./glove')
model.populate_embedding_layers_with_pretrained_word_vectors(glove_wordvec)

#%%----------------- Training -------------------------------------------------
lr, num_epochs = 0.001, 5
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
    eval_test_AUC=True,
    eval_each_batch=False
)

#%%--------------- Evaluate after training ------------------------------------
model.to('cpu')
X_test = test_data['padded_token_IDs'].to("cpu")
y_true_test = test_data['labels'].detach().numpy()
y_pred_test_ = model(X_test)
y_pred_class_test = torch.argmax(y_pred_test_, dim=1).detach().numpy()

y_pred_probability = y_pred_test_.detach().numpy()[:, 1]  # prob. of the POS class

from sklearn.metrics import accuracy_score, roc_auc_score

print('Accuracy = %.3f' % accuracy_score(y_true_test, y_pred_class_test))
print('AUC = %.3f' % roc_auc_score(y_true_test, y_pred_probability))
