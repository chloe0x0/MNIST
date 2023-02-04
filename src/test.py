import torch
import torchvision
import os
from train import Net, MODEL_PATH, device, norm, testing_loader
import matplotlib.pyplot as plt
from PIL import Image

# Using the trained model for inference

if __name__ == "__main__":
    assert(os.path.exists(MODEL_PATH))
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # batch of testing data
    it = iter(testing_loader)
    imgs, labels = next(it)

    # number of images to display
    TEST_LEN = 20

    # get predections tensor
    pred = model(imgs.view(imgs.shape[0], -1))
    pred = pred.data.max(1, keepdim=True)[1]

    fig = plt.figure(figsize=(TEST_LEN+5,4))
    for ix in range(TEST_LEN):
        # image to display
        img = imgs[ix].numpy().squeeze()
        # get prediction
        predicted = pred[ix].item()
        # get real value
        real = labels[ix]
        # add subplot
        ax = fig.add_subplot(2, TEST_LEN//2, ix+1, xticks=[], yticks=[])
        ax.axis('off')
        # show image in subplot
        ax.imshow(img, cmap="binary")
        # green if correct, red otherwise
        color = "green" if predicted==real else "red"
        ax.set_title(f"{predicted}", color=color)

    plt.show()