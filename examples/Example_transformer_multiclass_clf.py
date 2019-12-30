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
n = 200  # use just a small data set for training
texts_train = twenty_train['data'][:n]
labels_train = [int(label) for label in twenty_train['target']][:n]

texts_test = twenty_test['data'][:600]
labels_test = [int(label) for label in twenty_test['target']][:600]

assert(isinstance(texts_train, list))
assert(all([isinstance(text, str) for text in texts_train]))
assert(isinstance(labels_train, list))
assert(all([isinstance(label, int) for label in labels_train]))

assert(isinstance(texts_test, list))
assert(all([isinstance(text, str) for text in texts_test]))
assert(isinstance(labels_test, list))
assert(all([isinstance(label, int) for label in labels_test]))

#%%---------------- Prepare data ----------------------------------------------
model, tokenizer = dlu.transfomer_model_utils.get_model_and_tokenizer("DistilBERT")

max_length = 40  # because BERT models don't take sentences longer than 512 words

train_iter, _ = dlu.data_utils.create_text_data_iter(
    texts_train, labels_train, tokenizer, batch_size=64, max_length=max_length,
)

test_data, _ = dlu.data_utils.create_text_data_pack(
    texts_test, labels_test, tokenizer, max_length=max_length,
)

NUM_CLASSES = len(categories)

#%%--------------- Define model -----------------------------------------------
class TextClassifier(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(list(transformer.parameters())[-1].numel(), 256),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, NUM_CLASSES)
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

device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%--------------- Training ---------------------------------------------------
dlu.train_test_utils.train(
    train_iter=train_iter,
    model=clf,
    unpack_repack_fn=dlu.train_test_utils.unpack_training_data_for_transformer,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    device=device,
    num_epochs=5,
    test_data=None,  # don't evaluate during training (to save time and RAM)
    verbose_each_batch=True,
    eval_test_accuracy=True,
    eval_test_AUC=False,
)

#%%---------------- Evaluation ------------------------------------------------
model.to('cpu')
X_test = test_data['padded_token_IDs'].to("cpu")
y_true_test = test_data['labels'].detach().numpy()
masks = test_data['masks']

y_pred_test_ = clf(X_test, attention_mask=masks)
y_pred_class_test = torch.argmax(y_pred_test_, dim=1).detach().numpy()

from sklearn.metrics import accuracy_score

print('Accuracy = %.3f' % accuracy_score(y_true_test, y_pred_class_test))
