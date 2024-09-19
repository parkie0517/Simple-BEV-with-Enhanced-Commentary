import torch
import matplotlib.pyplot as plt

# Load the saved tensor
loaded_data = torch.load('./tensor.pt')

# Step 4: Extract and visualize each image
# Assuming loaded_data shape is (1, 6, 3, 224, 416)
batch, num_images, channels, height, width = loaded_data.shape

for i in range(num_images):
    img = loaded_data[0, i].permute(1, 2, 0)  # Rearrange the tensor from (C, H, W) to (H, W, C)

    plt.imshow(img.cpu().numpy())
    plt.title(f"Image {i + 1}")
    plt.show()
