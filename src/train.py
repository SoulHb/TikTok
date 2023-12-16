import segmentation_models_pytorch as smp
import sys
import torch
import argparse
import albumentations as A
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from torch import nn
sys.path.append('src')
from model.model import Unet
from dataset.dataset import TikTok
from config.config import *
torch.manual_seed(0)


def train_loop(model, loss, optimizer, train_loader, device):
    """
        Training loop for the model.

        Args:
            model: The neural network model.
            loss: The loss function.
            optimizer: The optimizer.
            train_loader: DataLoader for the training dataset.
            device (str): Device ('cuda' or 'cpu') to which the model and data should be moved.
        Returns: None
        """
    model.train()
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        predicted = model(X)
        Loss = loss(predicted, y)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        if i % 10 == 0:
            tp, fp, fn, tn = smp.metrics.get_stats(predicted>0.5, y.to(torch.int64), mode='binary', num_classes=1)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
            print(f"train_loss: {Loss.item()}, iou_score: {iou_score}, f1_score: {f1_score}, f2_score: {f2_score},"
              f"accuracy: {accuracy}, recall:{recall}")


def val_loop(model, val_loader, loss, device):
    """
       Validation loop for the model.

       Args:
           model: The neural network model.
           val_loader: DataLoader for the validation dataset.
           loss: The loss function.
           device (str): Device ('cuda' or 'cpu') to which the model and data should be moved.
       Returns: None
       """
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X = data[0].to(device)
            y = data[1].to(device)
            predicted = model(X)
            Loss = loss(predicted, y)
            if i % 10 == 0:
                tp, fp, fn, tn = smp.metrics.get_stats(predicted>0.5, y.to(torch.int64), mode='binary', num_classes=1)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
                accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
                print(f"test_loss: {Loss.item()}, iou_score: {iou_score}, f1_score: {f1_score}, f2_score: {f2_score},"
                      f"accuracy: {accuracy}, recall:{recall}")


def main(args):
    """
        Main function for training and saving the model.

        Args:
            args (dict): Dictionary containing training parameters and paths.
        Returns: None
        """
    transform = {'image': A.Compose(
        [A.Resize(736, 384), A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ToTensorV2(), ]),
                 'mask': A.Compose([A.Resize(736, 384), ToTensorV2(), ])}
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = args['epochs'] if args['epoch'] else EPOCHS
    lr = args['lr'] if args['lr'] else LR
    batch_size = args['batch_size'] if args['batch_size'] else BATCH_SIZE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Unet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    images_path = args['images_path']
    masks_path = args['masks_path']
    saved_model_path = args['saved_model_path'] if args['saved_model_path'] else SAVED_MODEL_FOLDER
    tiktok_dataset = TikTok(images_path, masks_path, transform=transform)
    train_ind, test_ind = train_test_split(list(range(len(tiktok_dataset))), shuffle=True, test_size=0.2)
    train_dataset = torch.utils.data.Subset(tiktok_dataset, train_ind)
    test_dataset = torch.utils.data.Subset(tiktok_dataset, test_ind)
    train_loader = torch.utils.data.DataLoader(train_dataset.dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=True)
    if saved_model_path:
       saved_model_path = f'{saved_model_path}/model'
    for i in range(epochs):
        train_loop(model, loss_fn, optimizer, train_loader, device)
        val_loop(model, test_loader, loss_fn, device)
        torch.save(model.state_dict(), f"{saved_model_path}/model_epoch_{i}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_path", type=str, help='Specify path to your result of train dataset')
    parser.add_argument("masks_path", type=str,
                        help='Specify path to your masks of train dataset')
    parser.add_argument("--saved_model_path", type=str,
                        help='Specify path for save model, where model folder will be created')
    parser.add_argument("--epochs", type=int,
                        help='Specify epoch for model training')
    parser.add_argument("--lr", type=float,
                        help='Specify learning rate')
    parser.add_argument("--batch_size", type=float,
                        help='Specify learning rate')
    args = parser.parse_args()
    args = vars(args)
    main(args)

