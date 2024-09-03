import torch
import torch.nn as nn
import argparse

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 2)  # A simple linear layer

    def forward(self, x):
        return self.fc(x)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Simple script to run NN on a specific GPU.')
parser.add_argument('-n', type=int, default=0, help='GPU number to use (e.g., 0, 1, 2, 3)')
args = parser.parse_args()

# Select the GPU device based on the argument
gpu_num = args.n

# Instantiate the neural network
model = SimpleNN()

# Check if CUDA is available and move the model to the GPU if possible
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_num}")
else:
    device = torch.device("cpu)")
    
model.to(device)

# Create a random input tensor
input_tensor = torch.randn(1, 10).to(device)

# Run the input through the neural network
output = model(input_tensor)

# Print the result and CUDA availability
print("CUDA is available:" if torch.cuda.is_available() else "CUDA is not available")
print("Output from the neural network:", output)
