import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets # MNIST Dataset

# A basic Neural Network for handwritten digit recognition 
# used as an introduction to PyTorch

# Hyperparameters
EPOCHS = 10
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
lr = 0.001 # Learning Rate

# Other constants
DATASET_PATH = "./MNIST/" # Root directory of where to store the MNIST dataset
LOG_INT = 5 # Interval to print log details during training

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
            # Output layer has 10 classes
            nn.Linear(64, 10),
            # Softmax layer
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''
        Basic forward pass through the network
        '''

        # Flatten the incoming image so it can be fed through the network
        x = x.flatten()
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
model = Net()

# Optimizer
# Will just use Adam
opt = optim.Adam(model.parameters(), lr=lr)

def train(epoch: int):
    '''
    A single training iteration of the model
    '''
    model.train()
    for idx, (feature, label) in enumerate(training_loader):
        # Need to ensure that the optimizer isnt accumulating gradients from previous iterations
        # zeroing the gradient is how this is achieved in PyTorch
        opt.zero_grad()
        # Predict the value
        y_hat = model(feature)
        # Compute the loss 
        loss = F.nll_loss(y_hat, label)
        # Compute the gradient of the loss function
        loss.backward()
        # Optimization step, Gradient Descent
        opt.step()
        
        # Log
        if (idx+1) % LOG_INT == 0:
            print(f"Training Epoch: {epoch} \t Loss: {loss.item():.6f}%")

# Test Function
def test():
    model.eval()
    loss = 0.0
    correct = 0
    # Context manager allows us to avoid accumulating gradients and messing with the computation graph
    with torch.no_grad():
        for feature, label in testing_loader:
            y_hat = model(feature)
            loss += F.nll_loss(y_hat, label, size_average=False).item()
            prediction = y_hat.data.max(1, keepdim=True)[1]
            correct += prediction.eq(label.data.view_as(prediction)).sum()
            
    # Average loss
    loss /= len(testing_loader.dataset)
    # Accuracy
    acc = correct / len(testing_loader.dataset)

    print(f"Testing Set: Avg Loss {loss} Accuracy {acc}")

if __name__ == "__main__":
    print("Training Network")
    for epoch in range(0, EPOCHS):
        train(epoch)
        test()