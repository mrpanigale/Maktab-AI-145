"""This file will train a Tiny Resnet """

#===========imports==========
from data_provider import provider,SEED
from run_experiment import run_experiment
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from pathlib import Path
import torch
from torch import nn

#===========Control_Randomness==========
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#===========Set-Device==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#===========Paths==========
data_root = Path("../data")

report_path = Path(__file__).parent.parent / "reports" / "csv"
report_path.mkdir(parents=True, exist_ok=True)

plots_path = Path(__file__).parent.parent / "reports" / "plots"
plots_path.mkdir(parents=True, exist_ok=True)
#===========BasicBlock==========
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        # ===========Convolutional Layers==========
        self.conv1 =nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.conv2 =nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        # ===========BatchNorms==========
        self.bn1 =nn.BatchNorm2d(out_channels)
        self.bn2 =nn.BatchNorm2d(out_channels)

        # ===========ActivationFunction==========
        self.relu =nn.ReLU()

        # ===========shortcut==========
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut=nn.Identity()

    # ===========forward-pass==========
    def forward(self,x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity

        x = self.relu(x)
        return x

#===========ResNet==========
class ResNet(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.stem =nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.stage1 = nn.Sequential(
            BasicBlock(16,16),
            BasicBlock(16,16),
        )

        self.stage2 = nn.Sequential(
            BasicBlock(16,32,stride=2),
            BasicBlock(32,32,stride=1),)

        self.head =nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32,10)
        )

    # ===========forward-pass==========
    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.head(x)

        return x


#===========Load-data==========
train_loader,val_loader,class_names = provider(data_path=data_root)

resnet = ResNet(len(class_names)).to(DEVICE)
#===========run-experiment==========
EPOCHS = 8


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)


history = run_experiment(
    model=resnet,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=DEVICE
    ,epoch=EPOCHS
)

#=========Save_Reports=========
csv_report = pd.DataFrame({
    "train_accuracy":history["train_accuracy"],
    "val_accuracy":history["val_accuracy"],
    "train_loss":history["train_loss"],
    "val_loss":history["val_loss"]
})

csv_report.to_csv(report_path/"Resnet.csv",index=False)

dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["train_confusion_matrix"][csv_report["train_loss"].argmin()],
    display_labels=class_names)
dsp.plot()
plt.savefig(plots_path/"Resnet.png")
plt.close()

print("Saved Successfully")