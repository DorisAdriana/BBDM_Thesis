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

# Dataset Class
class ImagePairDataset(Dataset):
    def __init__(self, base_dir, mode='train'):
        self.mode = mode
        self.base_dir = base_dir
        self.dir_A = os.path.join(base_dir, mode, 'A')
        self.filenames = sorted(os.listdir(self.dir_A))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_A_path = os.path.join(self.dir_A, self.filenames[idx])
        img_A = Image.open(img_A_path).convert('L')
        img_A = transforms.ToTensor()(img_A)
        img_A = transforms.Normalize(mean=[0.5], std=[0.5])(img_A)

        img_B_path = os.path.join(self.dir_A, '..', 'B', self.filenames[idx])
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
            nn.InstanceNorm2d(out_channels),  # Changed from BatchNorm2d to InstanceNorm2d
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),  # Changed from BatchNorm2d to InstanceNorm2d
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.k = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.q(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.k(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.v(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(2)
        self.attention = AttentionBlock(out_channels) if out_channels in [64, 128, 256, 512] else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout=0.0):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(mid_channels + out_channels, out_channels, dropout)
        self.attention = AttentionBlock(out_channels) if out_channels in [64, 128, 256, 512] else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(1, 64, dropout=0.1)
        self.enc2 = EncoderBlock(64, 128, dropout=0.1)
        self.enc3 = EncoderBlock(128, 256, dropout=0.2)
        self.enc4 = EncoderBlock(256, 512, dropout=0.2)
        self.bridge = ConvBlock(512, 1024, dropout=0.3)
        self.dec1 = DecoderBlock(1024, 512, 512, dropout=0.2)
        self.dec2 = DecoderBlock(512, 256, 256, dropout=0.2)
        self.dec3 = DecoderBlock(256, 128, 128, dropout=0.1)
        self.dec4 = DecoderBlock(128, 64, 64, dropout=0.1)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

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

# Mixed precision training scaler
scaler = torch.cuda.amp.GradScaler()

def train_model(model, train_loader, val_loader, epochs, architecture, model_name):
    checkpoint_dir = os.path.join(architecture, 'results/checkpoints', model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets, _ in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Mixed precision training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Average Training Loss: {total_loss / len(train_loader)}')
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets, _ in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        print(f'Epoch {epoch+1}, Average Validation Loss: {val_loss / len(val_loader)}')

        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch+1}.pth')

def predict_and_save(model, loader, model_path, output_dir):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, (inputs, _, filenames) in tqdm(loader, desc='Predicting'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = (outputs + 1) / 2
            outputs = outputs.cpu()

            for output, filename in zip(outputs, filenames):
                save_path = os.path.join(output_dir, filename)
                save_image(output, save_path)

if args.mode == 'train':
    train_model(model, train_loader, val_loader, config['epochs'], config['architecture'], config['model_name'])
elif args.mode == 'predict':
    if not args.model_path:
        raise ValueError("Model path must be provided for prediction mode.")
    predict_and_save(model, test_loader, args.model_path, args.output_dir)
