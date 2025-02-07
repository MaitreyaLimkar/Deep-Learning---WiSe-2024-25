import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()},
               f'checkpoints/checkpoint_{epoch:03d}.ckp')
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load(f'checkpoints/checkpoint_{epoch_n:03d}.ckp',
                     map_location='cuda' if self._cuda else 'cpu')
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        outputs = self._model(x)
        loss = self._crit(outputs, y)
        loss.backward()
        self._optim.step()
        return loss.item()
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        with t.no_grad():
            outputs = self._model(x)
            loss = self._crit(outputs, y)
        return loss.item(), outputs
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        self._model.train()
        total_loss = 0
        for x, y in self._train_dl:
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            loss = self.train_step(x, y)
            total_loss += loss
        return total_loss / len(self._train_dl)
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with t.no_grad():
            for x, y in self._val_test_dl:
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                loss, predict = self.val_test_step(x, y)
                total_loss += loss
                all_preds.append(predict.cpu())
                all_labels.append(y.cpu())
        avg_loss = total_loss / len(self._val_test_dl)
        predict_tensor = t.cat(all_preds, dim=0)
        labels_tensor = t.cat(all_labels, dim=0)
        predict_bin = (predict_tensor > 0.5).int()
        labels_bin = labels_tensor.int()
        f1 = f1_score(predict_bin, labels_bin, average='macro')
        print(f'Validation Loss: {avg_loss}, F1 Score: {f1}')
        return avg_loss, f1
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 

        train_losses, val_losses = [], []
        best_loss, patience = float('inf'), self._early_stopping_patience
        epoch = 0
        best_f1 = 0

        while True:
            # 1) Stop if we have reached the specified number of epochs
            if epoch == epochs:
                print("Reached maximum number of epochs.")
                break

            # 2) Training for one epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # 3) Validation step
            val_loss, val_f1 = self.val_test()
            val_losses.append(val_loss)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
            # Save the best model so far (based on F1)
                self.save_checkpoint(epoch)
            else:
                patience += 1

            # 5) Early stopping criterion
            if patience >= self._early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

            epoch += 1

        return train_losses, val_losses
                    
        
        
        
