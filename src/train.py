import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets # MNIST Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt

# A basic Neural Network for handwritten digit recognition 
# used as an introduction to PyTorch

# Epochs to train for
EPOCHS = 10
# Batch sizes
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 128
# Optimization parameters 
lr = 0.001      # Learning Rate
momentum = 0.9  # Momentum (used in SGD)

# Other constants
DATASET_PATH = "./MNIST/"           # Root directory of where to store the MNIST dataset
MODEL_PATH   = "./models/MNIST.pt"  # Path of where to store model checkpoints during training
SERIALIZE    = False                # Whether or not to save the model to a .pt file after training
LOG_INT      = 5                    # Interval to print log details during training
TRAIN        = False                # Whether or not to train a new model or load a pre-existing one

# Load Train and Test Data
# torchvision already provides a Dataset object for MNIST

# Normalization transform used when loading the dataset
# used as its own variable to reduce code re-use
norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        # These values are the global mean and std dev of the MNIST dataset (mean pixel val and std dev of pixel vals)
        # https://datascience.stackexchange.com/a/46235
        (0.1307,), (0.3081,)
    )
])

train_set = datasets.MNIST(
    # Where the training set will be stored
    DATASET_PATH, 
    # Load the training set
    train=True, 
    # Download the dataset to the path if it does not already exist
    download=True,
    # Normalize
    transform=norm
)

# Same idea for the testing set
test_set = datasets.MNIST(
    DATASET_PATH, 
    train=False, 
    download=True,
    transform=norm
)

# Create a Dataloader object for the training set 
training_loader = DataLoader(
    # Dataset object
    train_set, 
    # Batching parameters 
    batch_size=TRAIN_BATCH_SIZE,
    # Randomly shuffle
    shuffle=True
)

# Create a DataLoader for the testing set
testing_loader = DataLoader(
    test_set,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True
)

# Build Model
# Do not need to write a class for the network
# could just as easily have written model = nn.Sequential(...)
# may as well get used to writing classes for networks
class Net(nn.Module):
    def __init__(self):
        # Initialize the Pytorch Module object
        super(Net, self).__init__()
        # Will use a basic Feedforward MLP at the moment
        # Later write a CNN for better accuracy
        self.linear_relu = nn.Sequential(
            # each image is 28x28x1, so the input neuron dims is 28 * 28
            nn.Linear(28*28, 128),
            nn.ReLU(),
            # First Hidden Layer
            nn.Linear(128, 64),
            nn.ReLU(),
            # Second Hidden Layer
            nn.Linear(64, 64),
            nn.ReLU(),
            # Output layer has 10 classe
            nn.Linear(64, 10),
            # Softmax layer
            nn.LogSoftmax(dim=0)
        )

    def forward(self, x):
        '''
        Basic forward pass through the network
        '''

        # Feed the flattened image through the network
        y = self.linear_relu(x)

        return y

# Train model 

# Device to train on
# Will use GPU/ CUDA if possible, otherwise will use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Warning: Training model on CPU, could not find CUDA drivers.")

# Instantiate Model
model = Net().to(device)

# Optimizer
# Will use Stochastic Gradient Descent
opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(epoch: int):
    '''
    A single training iteration of the model
    '''
    model.train()
    for idx, (features, labels) in enumerate(training_loader):
        # set feature and label tensors to device
        features, labels = features.to(device), labels.to(device)
        # flatten features tensor
        features = features.view(features.shape[0], -1)
        # Need to ensure that the optimizer isnt accumulating gradients from previous iterations
        # zeroing the gradient is how this is achieved in PyTorch
        opt.zero_grad()
        # Predict the value
        y_hat = model(features)
        # Compute the loss 
        loss = F.nll_loss(y_hat, labels)
        # Compute the gradient of the loss function
        loss.backward()
        # Optimization step, Gradient Descent
        opt.step()
        
        # Log
        if idx % LOG_INT == 0:
            print(f"Training Epoch: {epoch} \t Loss: {loss.item():.6f}%")

# Test Function
def test():
    model.eval()
    loss = 0.0
    correct = 0
    # Context manager allows us to avoid accumulating gradients and messing with the computation graph
    with torch.no_grad():
        for features, labels in testing_loader:
            features, labels = features.to(device), labels.to(device)
            features = features.view(features.shape[0], -1)
            y_hat = model(features)
            loss += F.nll_loss(y_hat, labels, size_average=False).item()
            prediction = y_hat.data.max(1, keepdim=True)[1]
            correct += prediction.eq(labels.data.view_as(prediction)).sum()
            
    # Average loss
    loss /= len(testing_loader.dataset)
    # Accuracy
    acc = correct / len(testing_loader.dataset)

    print(f"Testing Set: Avg Loss {loss} Accuracy {acc}")

if __name__ == "__main__":
    if TRAIN:
        print("Training Network")
        for epoch in range(0, EPOCHS):
            train(epoch)
            test()
    
    # Serialize the model 
    if SERIALIZE and TRAIN:
        print(f"Saving trained model to {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)
    
    if not TRAIN:
        assert(os.path.exists(MODEL_PATH))
        print(f"Loading model from {MODEL_PATH}")
        model = Net().to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        '''
        it = iter(testing_loader)
        images, labels = next(it)
        images = images.view(images.shape[0], -1)
        y_hat = model(images)
        prediction = y_hat.data.max(1, keepdim=True)[1]
        print(prediction.reshape(labels.shape), '\n', labels)
        '''

        # Can now use for inference
        # load 0.png and see if it works
        zero = Image.open("./imgs/0.png")
        # Convert PIL Image to Tensor and flatten
        zero_tensor = norm(zero)
        zero_tensor = zero_tensor.view(zero_tensor.shape[0], -1)
        # Make a prediction
        y_hat = model(zero_tensor)
        class_ = y_hat.data.max(1, keepdim=True)[1].item()
        print(f"Image is a {class_}")
