"""This file will plot train&val curves for 3 experiments"""

#=============imports==============
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


report_path = Path(__file__).parent.parent /"reports"/"csv"
plots_path = Path(__file__).parent.parent /"reports"/"plots"
report_path.mkdir(parents=True, exist_ok=True)
plots_path.mkdir(parents=True, exist_ok=True)
# ========load_reports=========
regularized = pd.read_csv(report_path/"regularized_CNN.csv")
baseline = pd.read_csv(report_path/"FashionCNN.csv")
Augmented = pd.read_csv(report_path/"Aug_CNN.csv")

# ========make plots=========
x_axis = range(1,len(regularized)+1)
plt.figure(figsize=(12,5))
plt.plot(x_axis,regularized["train_loss"].values,linestyle="dashed",c='red',label="regularized")
plt.plot(x_axis,baseline["train_loss"].values,linestyle="dashed",c="blue",label="baseline")
plt.plot(x_axis,Augmented["train_loss"].values,linestyle="dashed",c="green",label="augmented")

plt.plot(x_axis,regularized["val_loss"].values,c='red',label="regularized")
plt.plot(x_axis,baseline["val_loss"].values,c="blue",label="baseline")
plt.plot(x_axis,Augmented["val_loss"].values,c="green",label="augmented")

plt.title("Loss Curves Train & Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
# ========save_plots=========

plt.savefig(plots_path/"loss_curves.png")
plt.close()

print("saved successfully")