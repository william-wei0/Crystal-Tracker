# train_resnet50_cosine.py
import os
import argparse
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

class PairAugmentedDataset(Dataset):
    """
    Expects:
        root/
          img_0001.jpg
          img_0002.jpg
          ...
    Returns two random augmentations of the same image.
    """
    def __init__(self, root: str, transform=None):
        self.transform = transform
        exts = ('*.jpg','*.jpeg','*.png','*.bmp')
        self.samples = []
        for e in exts:
            self.samples += glob(os.path.join(root, e))
        self.samples = sorted(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img  = Image.open(path).convert('RGB')
        # two independent random augmentations:
        a = self.transform(img)
        b = self.transform(img)
        return a, b


class EmbeddingNet(nn.Module):
    """
    ResNet50 backbone + projection to embedding_dim + L2 normalization.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        base = models.resnet50(pretrained=True)
        # strip off the average‐pool & FC
        modules = list(base.children())[:-2]  # keep convs
        self.backbone = nn.Sequential(*modules,
                                      nn.AdaptiveAvgPool2d((1,1)))
        in_feats = base.fc.in_features
        self.project = nn.Linear(in_feats, embedding_dim)

                # same preprocessing as training
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])

        self.training_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, imgs):
        if self.backbone.training:
            x = torch.stack([self.training_transform(im) for im in imgs]).to(next(self.backbone.parameters()).device)
        else:
            x = torch.stack([self.test_transform(im) for im in imgs]).to(next(self.backbone.parameters()).device)
        x = self.backbone(x)           # (B, 2048,1,1)
        x = x.flatten(1)               # (B, 2048)
        x = self.project(x)            # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1) # L2 normalize
        return x.cpu().numpy()


def train(args):
    # 1. transforms: random rotations, crops, flips
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomRotation((0,360)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        )
    ])

    # 2. dataset & loader
    dataset = PairAugmentedDataset(args.data_root, transform)
    loader  = DataLoader(dataset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.num_workers,
                         pin_memory=True,
                         drop_last=True)

    # 3. model, loss, optimizer
    model     = EmbeddingNet(embedding_dim=args.embed_dim)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CosineEmbeddingLoss(margin=0.0)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.1)

    # 4. training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0

        for a, b in loader:
            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            # targets = +1 for positive pairs
            target = torch.ones(a.size(0), device=device)

            optimizer.zero_grad()
            fa = model(a)  # (B, D)
            fb = model(b)  # (B, D)
            loss = criterion(fa, fb, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * a.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        print(f"[Epoch {epoch:02d}] cosine‐loss: {avg_loss:.4f}")

        # 5. checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                'epoch': epoch,
                'state_dict': model.state_dict()
            }
            path = os.path.join(args.out_dir, f"cosine_epoch{epoch}.pth")
            torch.save(ckpt, path)
            print(f" Saved checkpoint: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train ResNet50 embeddings with cosine loss"
    )
    p.add_argument('--data-root',
                   help="Folder of instance images",
                   default = r"path to individual crystal images folder")
    p.add_argument('--out-dir',   default='checkpoints_cosine',
                   help="Where to save checkpoints")
    p.add_argument('--epochs',    type=int, default=30)
    p.add_argument('--batch-size',type=int, default=32)
    p.add_argument('--lr',        type=float, default=1e-2)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--save-every',  type=int, default=5)
    p.add_argument('--embed-dim',   type=int, default=512,
                   help="Dimension of output embedding")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
