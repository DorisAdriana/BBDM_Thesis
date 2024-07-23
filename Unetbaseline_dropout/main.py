### CODE IS ADJUSTED TO RUN ON SEGMENTATIONS
### Run and see if improvement over baseline, if not: discard, if yes: include with multi-input as well

import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

# Argument Parser Setup
parser = argparse.ArgumentParser(description="Run the neural network training with configurable parameters.")
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, help='Batch size for training, validation, and testing.')
parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer.')
parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], help='Compute device to use ("auto", "cuda", or "cpu").')
parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Operation mode: "train" or "predict".')
parser.add_argument('--model_path', type=str, help='Path to the model file for prediction.')
parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save prediction outputs.')
args = parser.parse_args()

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.config)
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Override config with command line arguments if provided
if args.epochs:
    config['epochs'] = args.epochs
if args.batch_size:
    config['batch_size']['train'] = args.batch_size  # Assuming the same batch size for training, validation, and testing
    config['batch_size']['val'] = args.batch_size
    config['batch_size']['test'] = args.batch_size
if args.learning_rate:
    config['learning_rate'] = args.learning_rate
if args.device:
    config['device'] = args.device

# Define device based on command line or config
device_choice = args.device or config['device']
device = torch.device('cuda' if device_choice == 'cuda' and torch.cuda.is_available() else 'cpu')

class ImagePairDataset(Dataset):
    def __init__(self, base_dir, mode='train'):
        self.mode = mode
        self.base_dir = base_dir
        self.dir_A = os.path.join(base_dir, mode, 'A')
        self.dir_B = os.path.join(base_dir, mode, 'B')
        self.filenames = sorted(os.listdir(self.dir_A))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_A_path = os.path.join(self.dir_A, self.filenames[idx])
        img_A = Image.open(img_A_path).convert('L')
        img_A = transforms.ToTensor()(img_A)
        img_A = transforms.Normalize(mean=[0.5], std=[0.5])(img_A)

        # Replace .jpg with .png for images in folder B
        img_B_filename = self.filenames[idx].replace('.jpg', '.png')
        underscore_pos = [pos for pos, char in enumerate(img_B_filename) if char == '_']
        if len(underscore_pos) >= 2:
            img_B_filename = img_B_filename[:underscore_pos[2]] + img_B_filename[underscore_pos[2]+1:]
        img_B_path = os.path.join(self.dir_B, img_B_filename)
        img_B = Image.open(img_B_path).convert('L')
        img_B = transforms.ToTensor()(img_B)
        img_B = transforms.Normalize(mean=[0.5], std=[0.5])(img_B)

        return img_A, img_B, self.filenames[idx]

# Setup of the datasets and dataloaders
base_dir = config['base_dir']
train_dataset = ImagePairDataset(base_dir, config['mode']['train'])
val_dataset = ImagePairDataset(base_dir, config['mode']['val'])
test_dataset = ImagePairDataset(base_dir, config['mode']['test'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size']['train'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size']['val'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size']['test'], shuffle=False)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout=0.0):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(mid_channels + out_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(1, 16, dropout=0.1)  # Changed from 4-channel to 1-channel input
        self.enc2 = EncoderBlock(16, 32, dropout=0.1)
        self.enc3 = EncoderBlock(32, 64, dropout=0.2)
        self.enc4 = EncoderBlock(64, 128, dropout=0.2)
        self.bridge = ConvBlock(128, 256, dropout=0.3)
        self.dec1 = DecoderBlock(256, 128, 128, dropout=0.2)
        self.dec2 = DecoderBlock(128, 64, 64, dropout=0.2)
        self.dec3 = DecoderBlock(64, 32, 32, dropout=0.1)
        self.dec4 = DecoderBlock(32, 16, 16, dropout=0.1)
        self.final = nn.Conv2d(16, 1, kernel_size=1)  # Output channel is 1

    def forward(self, x):
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)
        b = self.bridge(p4)
        d1 = self.dec1(b, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)
        out = self.final(d4)
        return out

# Instantiate the model
model = UNet().to(device)

# Define loss function and optimizer
criterion = nn.L1Loss()  # MAE loss
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

def train_model(model, train_loader, val_loader, epochs, architecture, model_name):
    
    # Create model directory in checkpoints
    checkpoint_dir = os.path.join(architecture, 'results/checkpoints', model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # train loop    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets, _ in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Average Training Loss: {total_loss / len(train_loader)}')
        # eval loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets, _ in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        print(f'Epoch {epoch+1}, Average Validation Loss: {val_loss / len(val_loader)}')

        # Save the model at each epoch
        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch+1}.pth')

def predict_and_save(model, loader, model_path, output_dir):
    """
    Perform predictions using a trained model and save the output images using original filenames.

    Args:
    model (torch.nn.Module): The neural network model to use.
    loader (torch.utils.data.DataLoader): DataLoader containing the dataset for prediction.
    model_path (str): Path to the trained model file.
    output_dir (str): Directory to save prediction images.
    """
    # Load the model state
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prediction loop
    with torch.no_grad():
        for i, (inputs, _, filenames) in enumerate(tqdm(loader, desc='Predicting')):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Normalize outputs to [0, 1] if they were initially in the range [-1, 1]
            outputs = (outputs + 1) / 2
            outputs = outputs.cpu()

            # Save each output image, named after the original input filenames
            for output, filename in zip(outputs, filenames):
                save_path = os.path.join(output_dir, filename)
                save_image(output, save_path)

# Command-line handling logic
if args.mode == 'train':
    # Assuming train_model is correctly set up to handle training
    train_model(model, train_loader, val_loader, config['epochs'], config['architecture'], config['model_name'])
elif args.mode == 'predict':
    # Make sure model path and output directory are provided
    if not args.model_path:
        raise ValueError("Model path must be provided for prediction mode.")
    # Using the previously defined function
    predict_and_save(model, test_loader, args.model_path, args.output_dir)
