"""
This file will save some samples to check augmentation effect on them
"""

#=============imports=============
from pathlib import Path

from data_provider import provider,SEED
import matplotlib.pyplot as plt

import torch
from torchvision import transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("../data")
plots_path = Path(__file__).parent.parent / "reports"/"plots"
plots_path.mkdir(parents=True, exist_ok=True)

# =============denormalize function=============
def denormalize_gray(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


#=============load provider=============
train_loader,_,class_names = provider(data_path=data_root)

images,labels = next(iter(train_loader))
#=============augmentation=============
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.3),
])

aug_images = aug_transform(images)


#=============Num_Samples=============
num_samples = 10
fig, axes = plt.subplots(2,num_samples,figsize=(num_samples*2,10))

for i in range(num_samples):

    orig_img = images[i+10]
    aug_img = aug_images[i+10]

    orig_disp = denormalize_gray(orig_img).squeeze().cpu().numpy()
    aug_disp = denormalize_gray(aug_img).squeeze().cpu().numpy()


    axes[0,i].imshow(orig_disp,cmap="gray")
    axes[0,i].set_title(f"Org --> {class_names[labels[i+10]]} ")
    axes[0,i].axis('off')

    axes[1,i].imshow(aug_disp,cmap="gray")
    axes[1,i].set_title(f"Aug --> {class_names[labels[i+10]]}")
    axes[1,i].axis('off')

plt.tight_layout()
plt.savefig(str(plots_path / "aug_samples.png"))
plt.close()
print("Saved Successfully")
