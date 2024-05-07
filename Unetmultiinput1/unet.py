from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImagePairDataset(Dataset):
    def __init__(self, base_dir, mode='train', transform=None):
        """
        base_dir: Base directory containing the different folders of images.
        mode: Dataset mode, typically 'train', 'val', or 'test'.
        transform: Transformations to apply to each image.
        """
        self.mode = mode
        self.transform = transform
        # Define directories for each type of image
        self.dirs = {
            'original': os.path.join(base_dir, mode, 'A'),  # Images under subfolder 'A'
            'velx': os.path.join(base_dir + '_velx', mode),
            'vely': os.path.join(base_dir + '_vely', mode),
            'velz': os.path.join(base_dir + '_velz', mode)
        }
        # Use the original directory to define the length since it uses subfolder 'A'
        self.filenames = sorted(os.listdir(self.dirs['original']))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Fetch the 'original' image
        img_A_path = os.path.join(self.dirs['original'], self.filenames[idx])
        img_A = Image.open(img_A_path).convert('L')
        if self.transform:
            img_A = self.transform(img_A)

        # Fetch velocity images which do not have a subfolder 'A'
        imgs = [img_A]
        for key in ['velx', 'vely', 'velz']:
            img_path = os.path.join(self.dirs[key], self.filenames[idx])
            img = Image.open(img_path).convert('L')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        
        # Stack images along channel dimension to create a multi-channel input tensor
        img_tensor = torch.stack(imgs, dim=0)  # Shape will be [4, H, W]

        # Get the corresponding 'B' image (target image)
        img_B_path = os.path.join(base_dir, self.mode, 'B', self.filenames[idx])  # Adjust if needed
        img_B = Image.open(img_B_path).convert('L')
        if self.transform:
            img_B = self.transform(img_B)

        return img_tensor, img_B

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)  # Adjust normalization for 4 channels
])


print('Initialize dataloaders')
base_dir = 'data/slices_n98_s320x320_z88'
train_dataset = ImagePairDataset(base_dir, 'train', transform)
val_dataset = ImagePairDataset(base_dir, 'val', transform)
test_dataset = ImagePairDataset(base_dir, 'test', transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(mid_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(4, 16)  # adapted for 4 inputs
        self.enc2 = EncoderBlock(16, 32)
        self.enc3 = EncoderBlock(32, 64)
        self.enc4 = EncoderBlock(64, 128)
        
        self.bridge = ConvBlock(128, 256)
        
        self.dec1 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec3 = DecoderBlock(64, 32, 32)
        self.dec4 = DecoderBlock(32, 16, 16)
        
        self.final = nn.Conv2d(16, 1, kernel_size=1)  # Output channel is 1

    def forward(self, x):
        # Encoder
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)
        
        # Bridge
        b = self.bridge(p4)
        
        # Decoder
        d1 = self.dec1(b, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)
        
        # Final Convolution
        out = self.final(d4)
        return out

# Setup the device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the U-Net model
model = UNet().to(device)
print(model)

criterion = nn.L1Loss()  # MAE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, epochs):
    print('Starting training')
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, MAE Loss: {total_loss / len(train_loader)}')
        
        # Optional: Add validation logic here

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved!")

train_model(model, 1)  # Specify number of epochs

import os
import torch
from torchvision.utils import save_image
from PIL import Image

def predict_and_save(model, loader, output_dir='multi1predictions'):
    # Adjust path to include 'Unetbaseline'
    output_dir = os.path.join('Unetmultiinput1', output_dir)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    with torch.no_grad():
        for i, (inputs, _, filenames) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = (outputs + 1) / 2  # Normalize outputs if they are in range [-1, 1]
            outputs = outputs.cpu()

            # Iterate over each output and corresponding filename
            for output, filename in zip(outputs, filenames):
                # Remove the folder and extension from filename if needed
                filename = filename.split('/')[-1].replace('.jpg', '.png')
                save_path = os.path.join(output_dir, filename)
                save_image(output, save_path)

# Assuming you have the model loaded and a device set
model.load_state_dict(torch.load('Unetmultiinput1/model.pth'))
predict_and_save(model, test_loader)





