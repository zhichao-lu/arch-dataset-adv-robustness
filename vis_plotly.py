# %%
import numpy as np
import scipy
import pandas as pd
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from dataset import NASBenchR_CIFAR10_Dataset, Metric
from search_space import Arch
dataset = NASBenchR_CIFAR10_Dataset("data/cifar10_temp.jsonl")

root_path = Path("./fig")

# %%
df = pd.read_csv("data/cifar10.csv")
df["arch"] = pd.Series([eval(arch) for arch in df["arch"]])
df.sort_values(by="arch_id", inplace=True)
df.reset_index(drop=True, inplace=True)
df["arch_obj"] = pd.Series([f"{arch_id}: {str(Arch(*arch))}" for arch_id,arch in zip(df["arch_id"],df["arch"])])
df


# %%
metric_list = [
    Metric.MACS,
    Metric.PARAMS,
    Metric.VAL_BEST_CLEAN_ACC,
    Metric.VAL_BEST_PGD_ACC,
    Metric.VAL_BEST_CW_ACC,
]

metric_ba_list = [
    Metric.TEST_CLEAN_ACC,
    Metric.TEST_FGSM_ACC,
    Metric.TEST_FGSM_STABLE,
    Metric.TEST_PGD_ACC,
    Metric.TEST_PGD_STABLE,
    Metric.TEST_PGD_LIP,
    Metric.TEST_CW_ACC,
    Metric.TEST_CW_STABLE,
]

metric_aa_list = [
    Metric.TEST_AA_ACC
]
metric_val_list = [Metric.VAL_BEST_CLEAN_ACC,
                   Metric.VAL_BEST_PGD_ACC,
                   Metric.VAL_BEST_CW_ACC]

metric_ba_test_list = [
    Metric.TEST_CLEAN_ACC,
    Metric.TEST_FGSM_ACC,
    Metric.TEST_PGD_ACC,
    Metric.TEST_CW_ACC,
]

metric_list = [m.value for m in metric_list]
metric_list2 = ["train_last_acc",
                # "train_last_loss",
                "train_best_model_acc",
                # "train_best_model_loss",
                "val_best_model_acc",
                # "val_best_model_loss",
                "val_best_model_pgd"]
metric_ba_list = [m.value for m in metric_ba_list]
metric_aa_list = [m.value for m in metric_aa_list]

metric_val_list = [m.value for m in metric_val_list]
metric_ba_test_list = [m.value for m in metric_ba_test_list]

# %%
df_ba = df[df["test_clean_acc"].notna()]
fig = px.scatter(df_ba, x="macs", y="test_fgsm_acc",
           color="test_fgsm_acc", hover_data=["arch_obj"],
           color_continuous_scale=px.colors.sequential.matter
           )
fig.show()
# fig.show()
# %%
