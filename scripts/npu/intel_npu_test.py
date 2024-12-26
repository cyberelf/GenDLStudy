#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


import torch
from torch import nn
# import intel_npu_acceleration_library
# from intel_npu_acceleration_library.compiler import CompilerConfig
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import time

# Set random seeds
np.random.seed(0)
torch.manual_seed(0)

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


model = NeuralNetwork()
# compiler_conf = CompilerConfig(dtype=torch.float32, training=True)
# model = intel_npu_acceleration_library.compile(model, compiler_conf)

learning_rate = 1e-3
batch_size = 64

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    start_time = time.time()
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {t+1} completed in {epoch_duration:.2f} seconds\n")
print("Done!")

# Save the model
torch.save(model.state_dict(), "model.pth")


def inference_example(model, test_data):
    model.eval()
    # Load a random sample from the test dataset
    sample_idx = np.random.randint(len(test_data))
    sample, label = test_data[sample_idx]

    # Make a prediction
    with torch.no_grad():
        sample = sample.unsqueeze(0)  # Add batch dimension
        prediction = model(sample)
        predicted_label = prediction.argmax(1).item()

    # Visualize the result
    import matplotlib.pyplot as plt

    plt.imshow(sample.squeeze().numpy(), cmap="gray")
    plt.title(f"Predicted: {predicted_label}, Actual: {label}")
    plt.show()

# Call the inference function
inference_example(model, test_data)

