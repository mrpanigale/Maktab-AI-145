"""
This file trains a simple CNN model with augmentation Transform,on Fashion data and saves the model along with its reports.
"""

#=============imports=============
from pathlib import Path
from data_provider import provider,SEED
from run_experiment import run_experiment
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

import torch
import torch.nn as nn

#=============control_randomness=============
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("../data")
report_path = Path(__file__).parent.parent / "reports"/"csv"
plots_path = Path(__file__).parent.parent / "reports"/"plots"
report_path.mkdir(parents=True, exist_ok=True)
plots_path.mkdir(parents=True, exist_ok=True)
#=============Model=============
class FashionCNN(nn.Module):
    def __init__(self,n_classes:int):
        super().__init__()
        # =============attributes=============
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=16)
        self.conv2 =nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.fc = nn.Linear(1568,n_classes)

        self.n_classes = int(n_classes)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self,x):
        # =============Conv 1=============
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.pool(x)
        # =============Conv 2=============
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.pool(x)
        # ========Fully connected=========
        x = self.flatten(x)
        x = self.fc(x)

        return x


#=============load provider=============
train_loader,val_loader,class_names = provider(data_path=data_root,aug=True)
#=============cnn_obj=============
cnn = FashionCNN(n_classes=len(class_names)).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)


history = run_experiment(
    model=cnn,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=DEVICE
    ,epoch=3
)

#=========Save_Reports=========
csv_report = pd.DataFrame({
    "train_accuracy":history["train_accuracy"],
    "val_accuracy":history["val_accuracy"],
    "train_loss":history["train_loss"],
    "val_loss":history["val_loss"]
})

csv_report.to_csv(report_path/"Aug_CNN.csv",index=False)

dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["train_confusion_matrix"][csv_report["train_loss"].argmin()],
    display_labels=class_names)
dsp.plot()
plt.savefig(plots_path/"Aug_CNN.png")
plt.close()

print("Saved Successfully")
