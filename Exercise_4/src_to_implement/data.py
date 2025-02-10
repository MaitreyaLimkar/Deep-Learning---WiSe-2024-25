from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        # Initializing dataset with the data and mode parameters
        self.data = data
        self.mode = mode.lower()

        # Setting up different transformations based on the mode
        # Adding data augmentation for training mode by including random flips
        if self.mode == "train":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:  # Using simpler transformations for validation and other modes
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        # Returning the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Accessing the data row at the given index
        row = self.data.iloc[idx]

        # Loading the image from the file path
        image = imread(row["filename"])

        # Checking if need to convert grayscale images to RGB
        # Handling both (H, W) and (H, W, 1) formats
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image = gray2rgb(image)

        # Applying transformation pipeline to the image
        image = self.transform(image)

        # Extracting the binary labels for cracks and inactive areas
        crack_label = row["crack"]
        inactive_label = row["inactive"]
        label = torch.tensor([crack_label, inactive_label], dtype=torch.float)

        return image, label