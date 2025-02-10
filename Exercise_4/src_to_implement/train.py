import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Importing custom modules for dataset, model, and training
from data import ChallengeDataset
from model import ResNet
from trainer import Trainer

def main():
    # Loading data from CSV file
    df = pd.read_csv('data.csv', sep=';')

    # Splitting data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    # Creating dataset objects for training and validation
    train_data = ChallengeDataset(train_df, mode="train")
    val_data = ChallengeDataset(val_df, mode="val")

    # Setting up data loaders with batching and parallelization
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

    # Initializing the ResNet model
    model = ResNet()

    # Setting up binary cross-entropy loss function
    criterion = torch.nn.BCELoss()

    # Configuring Adam optimizer with learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Creating trainer instance with configuration
    trainer = Trainer(
        model=model,
        crit=criterion,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=val_loader,
        cuda=True,                    # Using GPU for training
        early_stopping_patience=80    # Stopping after 80 epochs without improvement
    )

    # Training the model and collecting metrics
    train_losses, val_losses, current_epoch = trainer.fit(epochs=40)

    # Printing training results
    print("Training finished!")
    print(current_epoch, " -" ,"Train losses:", train_losses)
    print("Validation losses:", val_losses)


if __name__ == "__main__":
    main()