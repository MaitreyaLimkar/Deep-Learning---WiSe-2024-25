from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode.lower()

        # Define transformations for train and validation modes.
        # Note: For training, we include a simple data augmentation (random horizontal flip).
        if self.mode == "train":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:  # For validation or any other mode, we use a fixed transformation.
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row from the dataframe.
        row = self.data.iloc[idx]

        # Read the image using skimage.io.imread.
        # Assumes that the dataframe has a column "filename" with the image path.
        image = imread(row["filename"])

        # Check if the image is grayscale. Many grayscale images have shape (H, W)
        # or shape (H, W, 1). In these cases, convert to RGB.
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image = gray2rgb(image)

        # Apply the transformation pipeline.
        image = self.transform(image)

        # Extract the label.
        # Assumes the dataframe has a column "label". Adjust as needed.
        crack_label = row["crack"]
        inactive_label = row["inactive"]
        label = torch.tensor([crack_label, inactive_label], dtype=torch.float)

        return image, label