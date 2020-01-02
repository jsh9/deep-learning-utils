# -*- coding: utf-8 -*-
"""
Multi-class classification using a transformer model.

Data set: 20 news groups dataset

(https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
"""
import torch
import transformers  # version >= 2.3.0
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
n = 1500  # use just a small data set for training
texts_train = twenty_train['data']#[:n]
labels_train = [int(label) for label in twenty_train['target']]#[:n]

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

#%%---------------- Prepare data ----------------------------------------------
model, tokenizer = dlu.transformer_model_utils.get_model_and_tokenizer("DistilBERT")

max_length = 40  # because BERT models don't take sentences longer than 512 words

train_iter, _ = dlu.data_utils.create_text_data_iter(
    texts_train, labels_train, tokenizer, batch_size=64, max_length=max_length,
)

test_iter, _ = dlu.data_utils.create_text_data_iter(
    texts_test, labels_test, tokenizer, batch_size=64, max_length=max_length,
)

NUM_CLASSES = len(categories)

#%%--------------- Define model -----------------------------------------------
class TextClassifier(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        embedding_size = list(transformer.parameters())[-1].numel()

        # Similar to https://github.com/huggingface/transformers/blob/629b22adcfe340c4e3babac83654da2fbd1bbf89/src/transformers/modeling_distilbert.py#L617
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(embedding_size, NUM_CLASSES)
        )

    def forward(self, token_IDs, attention_mask):
        last_hidden_states = self.transformer(
            token_IDs, attention_mask=attention_mask)[0]
        CLS_embedding = last_hidden_states[:, 0, :]  # extract [CLS] embedding
        output = self.classifier(CLS_embedding)
        return output

clf = TextClassifier(model)
optimizer = transformers.AdamW(
    [{"params": clf.transformer.parameters(), "lr": 2e-5, "correct_bias": False},
     {"params": clf.classifier.parameters(), "lr": 1e-3}]
)

#%%--------------- Training ---------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss()
unpack_repack_fn = dlu.train_test_utils.unpack_training_data_for_transformer

dlu.train_test_utils.train(
    train_iter=train_iter,
    model=clf,
    unpack_repack_fn=unpack_repack_fn,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    num_epochs=3,
    test_iter=test_iter,
    eval_each_batch=False,  # in order to save time
    verbose_each_batch=True,
    eval_test_accuracy=True,
    eval_test_AUC=False,
    eval_on_CPU=False,
)

#%%--------------- Evaluate after training ------------------------------------
test_scores, eval_txt = dlu.train_test_utils.eval_model(
    model=clf,
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
