import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:
    def __init__(self,
                 model,                        # Training this model
                 crit,                         # Using this loss function
                 optim=None,                   # Optimizing with this optimizer
                 train_dl=None,                # Training with this dataset
                 val_test_dl=None,             # Validating with this dataset
                 cuda=True,                    # Using GPU if True
                 early_stopping_patience=-1):  # Stopping early after this many epochs without improvement
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        # Moving model and criterion to GPU if requested
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        # Saving model state to checkpoint file
        t.save({'state_dict': self._model.state_dict()},
               f'checkpoints/checkpoint_{epoch:03d}.ckp')

    def restore_checkpoint(self, epoch_n):
        # Loading model state from checkpoint file
        ckp = t.load(f'checkpoints/checkpoint_{epoch_n:03d}.ckp',
                     map_location='cuda' if self._cuda else 'cpu')
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        # Converting model to ONNX format
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        _ = m(x)  # Running forward pass for shape inference
        t.onnx.export(
            m,                 # Exporting this model
            x,                 # Using this input
            fn,                # Saving to this file
            export_params=True,        # Including trained parameters
            opset_version=10,          # Using this ONNX version
            do_constant_folding=True,  # Optimizing constant operations
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

    def train_step(self, x, y):
        # Performing single training step
        self._optim.zero_grad()
        outputs = self._model(x)
        loss = self._crit(outputs, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):
        # Performing single validation/test step
        with t.no_grad():
            outputs = self._model(x)
            loss = self._crit(outputs, y)
        return loss.item(), outputs

    def train_epoch(self):
        # Training for one complete epoch
        self._model.train()
        total_loss = 0.0
        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            loss = self.train_step(x, y)
            total_loss += loss

        avg_loss = total_loss / len(self._train_dl)
        return avg_loss

    def val_test(self):
        # Running validation/test phase
        self._model.eval()
        total_loss = 0.0

        all_predicts = []
        all_labels = []

        with t.no_grad():
            for x, y in self._val_test_dl:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss, outputs = self.val_test_step(x, y)
                total_loss += loss

                # Collecting predictions and labels for metric calculation
                all_predicts.append(outputs.cpu())
                all_labels.append(y.cpu())

        avg_loss = total_loss / len(self._val_test_dl)

        # Computing F1 score
        predicts_tensor = t.cat(all_predicts, dim=0)
        labels_tensor = t.cat(all_labels, dim=0)

        predicts_temp = (predicts_tensor > 0.5).int()
        labels_temp = labels_tensor.int()

        f1 = f1_score(labels_temp, predicts_temp, average='macro')

        print(f"[Val] Loss: {avg_loss:.4f}, F1: {f1:.4f}")
        return avg_loss, f1

    def fit(self, epochs=-1):
        # Training the model for specified epochs or until early stopping
        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        best_f1 = 0.0
        patience_counter = 0
        current_epoch = 0

        while True:
            # Checking if reached maximum epochs
            if current_epoch == epochs:
                print("Reached maximum number of epochs.")
                break

            # Training for one epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Validating the model
            val_loss, val_f1 = self.val_test()
            val_losses.append(val_loss)

            # Tracking best performance by F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0

                self.save_checkpoint(current_epoch)
            else:
                patience_counter += 1

            # Checking early stopping criterion
            if patience_counter >= self._early_stopping_patience:
                print(f"Early stopping triggered at epoch {current_epoch}.")
                break

            current_epoch += 1

        return train_losses, val_losses, current_epoch