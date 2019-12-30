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
df_train = pd.read_csv(path_train, delimiter='\t', header=None)[:400]
df_test = pd.read_csv(path_test, delimiter='\t', header=None)[:500]
print('done.')

#%%---------------- Prepare data ----------------------------------------------
texts_train = list(df_train[0])
labels_train = list(df_train[1])

texts_test = list(df_test[0])
labels_test = list(df_test[1])

model, tokenizer = dlu.transformer_model_utils.get_model_and_tokenizer("DistilBERT")

train_iter, _ = dlu.data_utils.create_text_data_iter(
    texts_train, labels_train, tokenizer, batch_size=64,
)

test_data, _ = dlu.data_utils.create_text_data_pack(texts_test, labels_test, tokenizer)

NUM_CLASSES = len(set(labels_train))

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

#%%
dlu.train_test_utils.train(
    train_iter=train_iter,
    model=clf,
    unpack_repack_fn=dlu.train_test_utils.unpack_training_data_for_transformer,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    device=device,
    num_epochs=6,
    test_data=test_data,
    eval_each_batch=False,  # in order to save time and RAM
    verbose_each_batch=True,
    eval_test_accuracy=True,
    eval_test_AUC=True
)
