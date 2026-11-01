import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# from torchvision import datasets, transforms, models
from sklearn.metrics import balanced_accuracy_score
import sklearn
import imblearn
import os
import numpy as np
import random
import models
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
import numpy as np


from sklearn.metrics import confusion_matrix, balanced_accuracy_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--state', default=42, type=int,
                    help='seed for initializing training. ')


args1 = parser.parse_args()


def set_seeds(seed_value, use_cuda):
  np.random.seed(seed_value)  # cpu vars
  torch.manual_seed(seed_value)  # cpu  vars
  random.seed(seed_value)  # Python
  os.environ['PYTHONHASHSEED'] = str(seed_value) 
  if use_cuda:
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)  # gpu vars
      torch.backends.cudnn.deterministic = True  # needed
      torch.backends.cudnn.benchmark = False

set_seeds(args1.state, True)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import Counter

def calculate_metrics(pred, labels):
    
    # Convert probabilities to binary predictions
    binary_preds = torch.argmax(pred, dim=1)
    labels = labels
    
    # Balanced Accuracy (BACC)
    bacc = balanced_accuracy_score(labels.cpu().numpy(), binary_preds.cpu().numpy())

    # Geometric Mean
    geometric_mean = geometric_mean_score(labels.cpu().numpy(), binary_preds.cpu().numpy(), average = 'macro')
    
    # F1-score
    gt = labels.cpu().numpy()
    p = binary_preds.cpu().numpy()

    gt = gt.tolist()
    p = p.tolist()

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(gt,p)

    f1 = f1_score(gt, p, average='macro')

    return bacc, geometric_mean, f1, mcc#, recall_minority, precision_majority


def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs.to(device))
        
        # Calculate loss
        loss = criterion(outputs, labels.to(device))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    
    # Calculate average training loss
    avg_loss = total_loss / total_samples
    
    # Evaluate metrics on the training set
    with torch.no_grad():
        model.eval()
        all_preds = []
        all_labels = []
        
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            all_preds.append(outputs)
            all_labels.append(labels.to(device))
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

             
        # Calculate metrics
        bacc, geometric_mean, f1_score, mcc = calculate_metrics(all_preds, all_labels)
    
    # Print metrics
    print(f'Training Loss: {avg_loss:.4f} | BACC: {bacc:.4f} | Geometric Mean: {geometric_mean:.4f} | F1-Score: {f1_score:.4f} | MCC: {mcc:.4f}')

    return bacc, geometric_mean, f1_score, mcc


def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for inputs, labels in data_loader:
            # Forward pass
            outputs = model(inputs.to(device))
            
            # Calculate loss
            loss = criterion(outputs, labels.to(device))
            
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            all_preds.append(outputs)
            all_labels.append(labels)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        bacc, geometric_mean, f1_score, mcc = calculate_metrics(all_preds, all_labels)
    
    # Calculate average validation loss
    avg_loss = total_loss / total_samples
    
    # Print validation metrics
    print(f'Validation Loss: {avg_loss:.4f} | BACC: {bacc:.4f} | Geometric Mean: {geometric_mean:.4f} | F1-Score: {f1_score:.4f} | MCC: {mcc:.4f}')

    return bacc, geometric_mean, f1_score, mcc

image_size = 32
data_transforms = {
    'train': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
}

dataset = 'mnist' 

#provide the folder name to classify containing the train folder
folder = "./data/"+dataset+"/supcon0.01_5_41"
data_dir = [folder]

for i in range(len(data_dir)):

    print("\n\n dataset:",data_dir[i],"\n\n")
    train_dataset = datasets.ImageFolder(os.path.join(data_dir[i], 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join("./data/"+dataset+"/validation", 'val'), data_transforms['val'])
    val_dataset = datasets.ImageFolder(os.path.join("./data/"+dataset+"/validation", 'val'), data_transforms['val'])


    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12)
    val_dataloder = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=12)


    # model = model.to(device)
    model = models.__dict__['resnet32'](num_classes=10, use_norm=False).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)


    best_val_bacc = 0.0  # Initialize the best validation BACC
    best_epoch = 0

    # Assuming you have already defined your model, train_loader, val_loader, criterion, and optimizer
    num_epochs = 100  # Change this to the desired number of epochs
    best_validation_acc = 0.0
    best_gmean = 0.0
    best_f1 = 0.0
    best_mcc = 0.0
    

    f = open("./results/"+dataset+"/metrics_resnet32_"+str(args1.state)+".txt","a")
    f.write("\n"+"metrics of:"+data_dir[i]+"\n")
    f.write("bal_acc,mathews,f1,geometric")
    for epoch in range(num_epochs):

        # Train the model
        bacc, geometric_mean, f1, mcc = train(model, train_dataloder, optimizer, criterion)
        
        # Validate the model
        bacc, geometric_mean, f1, mcc = validate(model, val_dataloder, criterion)

        # Update best validation accuracy and save the model if it improved
        if bacc > best_validation_acc:
            best_validation_acc = bacc
            best_gmean =  geometric_mean
            best_f1 =  f1
            best_mcc =  mcc
            torch.save(model.state_dict(), data_dir[i]+'.pth')  # Change the filename as needed
            print(f'Saved model with improved validation accuracy: {best_validation_acc:.4f}')
        print(f'Best validation accuracy: {best_validation_acc:.4f}')
    f.write("\n"+str(best_validation_acc)+","+str(best_mcc)+","+str(best_f1)+","+str(best_gmean))
    f.close()
    print('Training complete.')