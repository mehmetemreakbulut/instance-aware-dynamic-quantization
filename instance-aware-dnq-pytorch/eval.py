import os
import argparse
import torch
import time
import torch.nn as nn
from torch.utils import checkpoint
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.resnet import ResNet, resnet18
from src.loss import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default="./res18.pth", help='Checkpoint file path')
parser.add_argument('--val_dataset_path', type=str, default="../../data/imagenet/val/", help='Dataset path')
parser.add_argument('--train_dataset_path', type=str, default="../../data/imagenet/val/", help='Dataset path')
parser.add_argument('--device', type=str, default="cuda", help='Run device target')
parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing (default: 0.1)')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N', help='number of label classes (default: 1000)')
parser.add_argument('--tar_bit', type=int, default=4, help="Target bit quantization" )
parser.add_argument('--alpha', type=float, default=0.05, help="Alpha value of regularization" )
args = parser.parse_args()

def create_dataset_val(val_data_url, batch_size=256, workers=8):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(val_data_url, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return dataloader

def validate(model, dataloader, criterion, device, tar_bit):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    #bit list is {tar_bit-1, tar_bit, tar_bit+1}
    bit_list = torch.Tensor([tar_bit-1, tar_bit, tar_bit+1]).to(device)
    bit_list = bit_list.cuda()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, one_hot = model(inputs)

            #quantization code
            avg_bit = torch.mean(torch.sum(one_hot*bit_list, 2))
            avg_bit = avg_bit.float()

            loss = criterion(outputs, targets) + (0.05 * torch.clamp(avg_bit - tar_bit, min=0))

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def train(train_loader, model, criterion, optimizer, epoch, scheduler, args, tar_bit):
    # switch to train mode
    model.train()
    bit_list = torch.Tensor([args.tar_bit-1,args.tar_bit,args.tar_bit+1])
    #bit_list = bit_list.cuda()
    for i, (images, target) in enumerate(train_loader):
        print(images.size())
        print(target.size())

        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)

        # compute output
        start = time.time()
        output, one_hot = model(images)
        end = time.time()
        print("Time in seconds format: ", end - start)

        bit = (torch.sum(one_hot*bit_list)/ one_hot.size(0))
        loss = criterion(output, target) + args.alpha * torch.clamp(bit-tar_bit, min=0.0)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




if __name__ == '__main__':
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")
    # Create model
    network = resnet18(num_classes=args.num_classes)

    # Load pre-trained checkpoint
    if False:
        #checkpoint = torch.load(args.checkpoint_path, map_location=device)

        #checkpoint = torch.load(args.checkpoint_path)
        #network.load_state_dict(checkpoint['state_dict'])
        pass


    network = network.to(device)

    # Define loss function
    criterion = CrossEntropySmooth(smooth_factor=args.smoothing, num_classes=args.num_classes)

    optimizer  = SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Create train dataset
    train_dataset = create_dataset_val(val_data_url=args.train_dataset_path)
    print(torch.cuda.is_available())
    for epoch in range(1, 100):
        train(train_loader=train_dataset, model=network, criterion=criterion, optimizer=optimizer, epoch=epoch, scheduler=None, args=args, tar_bit=args.tar_bit)
        print(f"Epoch: {epoch}")


    #SAVE MODEL
    torch.save(network.state_dict(), args.checkpoint_path)


    # Create val dataset
    val_dataset = create_dataset_val(val_data_url=args.val_dataset_path)


    print("============== Starting Validation ==============")
    val_loss, val_accuracy = validate(network, val_dataset, criterion, device, args.tar_bit)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print("============== End Validation ==============")
