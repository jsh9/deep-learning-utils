import time
import torch
from typeguard import typechecked
from typing import Tuple, Dict, Optional, Any

TensorTensorTensor = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

#%%----------------------------------------------------------------------------
@typechecked
def retrieve_X_y_masks(data: Dict) -> TensorTensorTensor:
    X = data['padded_token_IDs']
    y = data['labels']
    masks = data['masks']
    return X, y, masks

#%%----------------------------------------------------------------------------
@typechecked
def train_transformer(
        train_iter: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: str = "cpu",
        test_data: Optional[Dict[str, torch.Tensor]] = None,
        verbose_each_batch: bool = True,
        verbose_each_epoch: bool = True):
    """
    Train a transformer-based model.

    Parameters
    ----------
    train_iter : torch.utils.data.DataLoader
        Training data iteration object. Each item in the iterator must be a
        dictionary with three keys: "padded_token_IDs", "labels", and "masks",
        and the corresponding values being PyTorch tensors.
    model : torch.nn.Module
        A neural network model. Its first layer must be a transformer model
        (as in the ``transformers`` library).
    loss_fn : torch.nn.modules.loss._Loss
        Loss function.
    optimizer : torch.optim.Optimizer
        The optimizer to update the model weights.
    num_epochs : int
        Number of epochs to train the model.
    device : str
        The device to do the training on. It can be "cpu", or "cuda", or other
        valid names (if you have multiple GPUs). Default value is "cpu".
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
        optimizer = transformers.AdamW(
            [{"params": clf.transformer.parameters(), "lr": 2e-5, "correct_bias": False},
             {"params": clf.classifier.parameters(), "lr": 1e-3}]
        )
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
            model.zero_grad()
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

#%%----------------------------------------------------------------------------
@typechecked
def train(
        train_iter: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: str = "cpu",
        other_args_to_model: Dict[str, Any] = dict(),
        test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        verbose_each_batch: bool = True,
        verbose_each_epoch: bool = True):
    """
    Train a generic PyTorch model.

    Parameters
    ----------
    train_iter : torch.utils.data.DataLoader
        Training data iteration object. Each item in the iterator must be a
        tuple of two PyTorch tensors ("`X`" and "`y`" respectively).
    model : torch.nn.Module
        A neural network model.
    loss_fn : torch.nn.modules.loss._Loss
        Loss function.
    optimizer : torch.optim.Optimizer
        The optimizer to update the model weights.
    num_epochs : int
        Number of epochs to train the model.
    device : str
        The device to do the training on. It can be "cpu", or "cuda", or other
        valid names (if you have multiple GPUs). Default value is "cpu".
    other_args_to_model : Dict[str, Any]
        Other arguments to pass to the ``model``'s ``forward()`` method. You
        need to pack them into a dictionary, whose keys are the argument names
        and values are argument values. If it is an empty dict, it means that
        no additional arguments will be passed to ``forward()``.
    test_data : Tuple[torch.Tensor, torch.Tensor] or ``None``
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
        import deep_learning_utils as dlu

        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(9, 4)
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(p=0.2)
                self.layer2 = torch.nn.Linear(4, 1)

            def forward(self, input_):
                x = self.layer1(input_)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.layer2(x)
                return x

        clf = Classifier()

        optimizer = torch.optim.Adam(
            [{"params": clf.layer1.parameters(), "lr": 0},
             {"params": clf.layer2.parameters(), "lr": 1e-3}]
        )

        X = torch.rand(124, 9)
        y = torch.rand(124, 1)

        dataset = torch.utils.data.TensorDataset(X, y)
        train_iter = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        loss_fn = torch.nn.MSELoss()

        dlu.train_test_utils.train(train_iter, model=clf, loss_fn=torch.nn.MSELoss(),
                                   optimizer=optimizer, num_epochs=20)


    Notes
    -----
    Modified from:
    https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/25d5e050179d32310821cb9b6ea2a0905ab62e00/code/d2lzh_pytorch/utils.py#L696
    """
    if test_data is not None:
        X_test = test_data[0].to(device)
        y_test = test_data[1].to(device)
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
            X = training_data[0].to(device)
            y = training_data[1].to(device)
            y_pred = model(X, **other_args_to_model)
            loss = loss_fn(y_pred, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            batch_count += 1

            if test_data is not None:
                y_pred_on_test = model(X_test, **other_args_to_model)
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

