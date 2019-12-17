import time
import torch
from typeguard import typechecked
from typing import Tuple, Dict, Optional

ThreeTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

@typechecked
def retrieve_X_y_masks(data: Dict) -> ThreeTensors:
    X = data['padded_token_IDs']
    y = data['labels']
    masks = data['masks']
    return X, y, masks

# @typechecked
def train_transformer(
        train_iter: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        device: str,
        num_epochs: int,
        test_data: Optional[Dict[str, torch.Tensor]] = None,
        verbose_each_batch: bool = True,
        verbose_each_epoch: bool = True):
    """
    Train a transformer-based model.

    Parameters
    ----------
    train_iter :
        Training data iteration object. It must be the output of a
        ``DataLoader``.
    model : torch.nn.Module
        A neural network model.
    loss_fn : torch.nn.modules.loss._Loss
        Loss function.
    optimizer : torch.optim.Optimizer
        The optimizer to update the model weights.
    device : str
        The device to do the training on. It can be "cpu", or "cuda", or other
        valid names (if you have multiple GPUs).
    num_epochs : int
        Number of epochs to train the model.
    test_data : Dict[str, torch.Tensor] or ``None``
        Test data to evaluate the model performance on. Setting it to ``None``
        means you do not want to evaluate test loss at every batch. Passing
        a ``test_data`` can slow down the whole training process.
    verbose_each_batch : bool
        Whether to print loss values at each batch.
    verbose_each_epoch : bool
        Whether to print loss values at each epoch.

    Example
    -------
    .. code-block:: python

        import torch
        import transformers
        import deep_learning_utils as dlu

        model, tokenizer = dlu.transfomer_model_utils.get_model_and_tokenizer("DistilBERT")

        train_texts = ["This movie is brilliant.", "This movie is mediocre.", "Great"]
        train_labels = [1, 0, 1]
        train_iter = dlu.data_utils.create_data_iter(train_texts, train_labels, tokenizer)

        test_texts = ["Just so-so.", "Best I have every seen."]
        test_labels = [0, 1]
        test_data = dlu.data_utils.pack_texts_and_labels(test_texts, test_labels, tokenizer)

        class TextClassifier(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(list(transformer.parameters())[-1].numel(), 256),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(256, 2)
                )

            def forward(self, token_IDs, attention_mask):
                last_hidden_states = self.transformer(
                    token_IDs, attention_mask=attention_mask)[0]
                CLS_embedding = last_hidden_states[:, 0, :]  # extract [CLS] embedding
                output = self.classifier(CLS_embedding)
                return output

        clf = TextClassifier(model)
        optimizer = transformers.AdamW(clf.parameters(), lr=2e-5, correct_bias=False)
        dlu.train_test_utils.train_transformer(
            train_iter, model=clf, loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device='cpu',
            num_epochs=4,
            test_data=test_data,
            verbose_each_batch=True)

    Notes
    -----
    Modified from:
    https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/25d5e050179d32310821cb9b6ea2a0905ab62e00/code/d2lzh_pytorch/utils.py#L696
    """
    if test_data is not None:
        X_test = test_data['padded_token_IDs'].to(device)
        y_test = test_data['labels'].to(device)
        masks_test = test_data['masks'].to(device)
    # END IF

    model = model.to(device)
    print("Training on:", device)
    for epoch in range(num_epochs):
        if verbose_each_epoch and verbose_each_batch:
            print('Epoch %d:' % epoch)
        # END IF

        train_loss_sum = 0.0
        test_loss_sum = 0.0
        batch_count = 0
        t0 = time.time()
        for training_data in train_iter:  # batch by batch
            X, y, masks = retrieve_X_y_masks(training_data)
            X = X.to(device)
            y = y.to(device)
            masks = masks.to(device)
            y_pred = model(X, attention_mask=masks)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            batch_count += 1

            if test_data is not None:
                y_pred_on_test = model(X_test, attention_mask=masks_test)
                test_loss = loss_fn(y_pred_on_test, y_test)
                test_loss_sum += test_loss.cpu().item()
            # END IF

            if verbose_each_batch:
                if test_data is None:
                    print('Training loss: %.4f' % loss.item())
                else:
                    print('Training loss: %.4f, test loss: %.4f' \
                          % (loss.item(), test_loss))
                # END IF-ELSE
            # END IF
        # END FOR

        dt = time.time() - t0
        training_loss_this_epoch = train_loss_sum / batch_count
        test_loss_this_epoch = test_loss_sum / batch_count
        if verbose_each_epoch:
            if test_data is not None:
                print('Epoch %d. Training loss: %.4f, test loss: %.4f (time %.1f sec)' \
                      % (epoch, training_loss_this_epoch, test_loss_this_epoch, dt))
            else:
                print('Epoch %d. Training loss: %.4f (time %.1f sec)' \
                      % (epoch, training_loss_this_epoch, dt))
            # END IF-ELSE
        # END IF
        if verbose_each_epoch and verbose_each_batch:
            print('---------------------')
        # END IF
    # END FOR
