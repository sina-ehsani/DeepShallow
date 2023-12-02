# custom weighted loss function:
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    return


class WeightedMSELoss(nn.Module):
    def __init__(self, weight=10.0):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        # Create a mask with 'weight' for non-zero values and 1 for zero values
        mask = target.clone()
        mask[mask != 0] = self.weight
        mask[mask == 0] = 1.0

        # Compute the element-wise MSE loss
        loss = (output - target) ** 2

        # Weight the losses
        weighted_loss = loss * mask

        # Return the mean loss
        return weighted_loss.mean()


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs,
    validation_interval,
    patience,
    scheduler=None,
    print_losses=True,
):
    train_losses = []
    val_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=print_losses)

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        if (epoch) % validation_interval == 0:
            if print_losses:
                print("Epoch {}/{}".format(epoch + 1, num_epochs))
                print("-" * 10)

        # training phase
        model.train()
        running_loss = 0.0

        for data in train_loader:
            # check if the loader provides two inputs or just one
            if len(data) == 3:
                inputs1, inputs2, labels = data
                inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            else:
                inputs1, labels = data
                inputs1 = inputs1.to(device)

            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            if len(data) == 3:
                outputs = model(inputs1, inputs2)
            else:
                outputs = model(inputs1)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * len(labels)

        if scheduler:
            scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # validation phase
        if epoch % validation_interval == 0:
            if print_losses:
                print("Train Loss: {:.4f}".format(epoch_loss))
            model.eval()
            running_loss = 0.0

            for data in val_loader:
                # check if the loader provides two inputs or just one
                if len(data) == 3:
                    inputs1, inputs2, labels = data
                    inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
                else:
                    inputs1, labels = data
                    inputs1 = inputs1.to(device)

                labels = labels.to(device)

                # forward, no need to track history
                with torch.no_grad():
                    if len(data) == 3:
                        outputs = model(inputs1, inputs2)
                    else:
                        outputs = model(inputs1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * len(labels)

            epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append((epoch, epoch_loss))
            if print_losses:
                print("Validation Loss: {:.4f}".format(epoch_loss))

            early_stopping(epoch_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load("checkpoint.pt"))

    if not print_losses:
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("Last Train Loss: {:.4f}".format(train_losses[-1]))
        print("Best Validation Loss: {:.4f}".format(early_stopping.val_loss_min))

    return model, train_losses, val_losses, early_stopping.val_loss_min


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
