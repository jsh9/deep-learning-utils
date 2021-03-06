# -*- coding: utf-8 -*-
"""
Binary classification using a transformer model.

Data set: IMDb sentiment, 2-class (SST-2)
"""
import torch
import transformers  # version >= 2.3.0
import pandas as pd
import deep_learning_utils as dlu

#%%-------------------- Load IMDB sentiment data ------------------------------
print('Loading data... ', end='')
path_train = './data/SST-2/train.tsv' # 6920 records in total
path_test = './data/SST-2/test.tsv'  # 1821 records in total
df_train = pd.read_csv(path_train, delimiter='\t', header=None)[:1000]
df_test = pd.read_csv(path_test, delimiter='\t', header=None)
print('done.')

#%%---------------- Prepare data ----------------------------------------------
texts_train = list(df_train[0])
labels_train = list(df_train[1])

texts_test = list(df_test[0])
labels_test = list(df_test[1])

model, tokenizer = dlu.transformer_model_utils.get_model_and_tokenizer("DistilBERT")

train_iter, _ = dlu.data_utils.create_text_data_iter(
    texts_train, labels_train, tokenizer, batch_size=32, max_length=40,
)

test_iter, _ = dlu.data_utils.create_text_data_iter(
    texts_test, labels_test, tokenizer, batch_size=32, max_length=40,
)

NUM_CLASSES = len(set(labels_train))

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

#%%----------------- Training -------------------------------------------------
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
    num_epochs=4,
    test_iter=test_iter,
    eval_each_batch=False,  # in order to save time
    verbose_each_batch=True,
    eval_test_accuracy=True,
    eval_test_AUC=True,
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
    eval_AUC=True,
    eval_R2=False,
    verbose=True
)
