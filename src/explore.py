import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets # MNIST Dataset
import matplotlib.pyplot as plt
import numpy as np

# Basic exploration/ visualization of the dataset

# Bring in the datasets
# assumption is that they are already installed

# Re-define some constants from train.py rather than importing from it

TRAIN_BATCH_SIZE = 64
DATASET_PATH = "./MNIST/"

norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])

train_set = datasets.MNIST(
    DATASET_PATH, 
    train=True, 
    download=True,
    transform=norm
)

training_loader = DataLoader(
    train_set, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)

if __name__ == "__main__":
    it = iter(training_loader)
    images, labels = next(it)

    fig = plt.figure(figsize=(3, 3))
    fig.tight_layout(pad=15.0)
    imgs = 15
    for ix in range(1, imgs+1):
        plt.subplot(3, 5, ix)
        plt.axis('off')
        plt.imshow(images[ix].numpy().squeeze(), cmap="binary")
        print(images.view(images.shape[0], -1)[ix].shape, images.shape)
        plt.title(labels[ix].data.item())

    plt.show()
