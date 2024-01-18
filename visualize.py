# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import NASBenchR_CIFAR10_Dataset, Metric
from search_space import Arch
dataset = NASBenchR_CIFAR10_Dataset("data/cifar10_temp.jsonl")
# %%
complete_train_archs = list(dataset.dataset.keys())
complete_train_archs = set(complete_train_archs)
complete_ba_archs = [
    arch for arch in complete_train_archs
    if dataset.query(arch).train_records[0].best_model_attack_acc.Linf_fgsm_acc is not None
]
complete_ba_archs = set(complete_ba_archs)
complete_aa_archs = [
    arch for arch in complete_train_archs
    if dataset.query(arch).train_records[0].best_model_attack_acc.Linf_aa_acc is not None
]
complete_aa_archs = set(complete_aa_archs)

complete_both_archs = list(complete_ba_archs & complete_aa_archs)
print(f"complete_train_archs: {len(complete_train_archs)}")
print(f"complete_ba_archs: {len(complete_ba_archs)}")
print(f"complete_aa_archs: {len(complete_aa_archs)}")
print(f"complete_both_archs: {len(complete_both_archs)}")
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

# %%
dict_list = []
for arch in complete_train_archs:
    record = dataset.query(arch)
    rec_dict = dict(
        arch_id=record.arch_id,
        arch=record.arch.to_tuple()
    )
    for metric in metric_list:
        value = dataset.get_metric(record, metric)
        rec_dict[metric.value] = value

    if arch in complete_ba_archs:
        for metric in metric_ba_list:
            value = dataset.get_metric(record, metric)
            rec_dict[metric.value] = value

    if arch in complete_aa_archs:
        for metric in metric_aa_list:
            value = dataset.get_metric(record, metric)
            rec_dict[metric.value] = value

    dict_list.append(rec_dict)

df = pd.DataFrame.from_dict(dict_list)
df
# %%
df.to_csv("data/cifar10.csv", index=False)

# %%
df = pd.read_csv("data/cifar10.csv")
df["arch"] = pd.Series([eval(arch) for arch in df["arch"]])
df.sort_values(by="arch_id", inplace=True)
df.reset_index(drop=True, inplace=True)
df
# %%
# ============= Plotting =============
# %%


def plot_scatter(x_metric, y_metric, df_new=None, **kwargs):
    if df_new is None:
        df_new = df
    df_temp = df_new[df_new[y_metric].notna()]
    fig = plt.figure(figsize=(10, 10), dpi=300)
    default_opts = dict(
        alpha=0.5, size=1, legend=False, ec=None
    )
    default_opts.update(kwargs)

    sns.scatterplot(data=df_temp, x=x_metric, y=y_metric,
                    hue=y_metric, **default_opts)
    plt.show()


# %%
fig = plt.figure(figsize=(10, 10), dpi=300)

df1 = df[df["test_clean_acc"].notna()]
df1 = df1[df1["test_clean_acc"] > 84]

sns.scatterplot(data=df1, x="params", y="test_clean_acc",
                hue="test_clean_acc", alpha=0.5, size=1, legend=False, ec=None)
plt.show()


# %%
metric_val_list = [Metric.VAL_BEST_CLEAN_ACC,
                   Metric.VAL_BEST_PGD_ACC,
                   Metric.VAL_BEST_CW_ACC]

metric_ba_test_list = [
    Metric.TEST_CLEAN_ACC,
    Metric.TEST_FGSM_ACC,
    Metric.TEST_PGD_ACC,
    Metric.TEST_CW_ACC,
]
# %%
x_metric = Metric.PARAMS.value
for metric in metric_val_list:
    y_metric = metric.value
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)

# %%
x_metric = Metric.MACS.value
for metric in metric_val_list:
    y_metric = metric.value
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)

# %%
x_metric = Metric.PARAMS.value
for metric in metric_ba_test_list:
    y_metric = metric.value
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)
# %%
x_metric = Metric.MACS.value
for metric in metric_ba_test_list:
    y_metric = metric.value
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)
# %%
x_metric = Metric.PARAMS.value
for metric in metric_aa_list:
    y_metric = metric.value
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)
# %%
x_metric = Metric.MACS.value
for metric in metric_aa_list:
    y_metric = metric.value
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)
# %%


def plot_reg(x_metric, y_metric, df_new=None, **kwargs):
    if df_new is None:
        df_new = df
    df_temp = df_new[df_new[y_metric].notna()]
    fig = plt.figure(figsize=(10, 10), dpi=300)
    default_opts = dict(
        scatter_kws=dict(alpha=0.5, s=2, color='black'),
        line_kws=dict(color="red", alpha=0.3),
        ci=95, robust=True

    )
    default_opts.update(kwargs)

    sns.regplot(data=df_temp, x=x_metric, y=y_metric,  **default_opts)
    plt.show()


# %%
x_metric = Metric.VAL_BEST_CLEAN_ACC.value
y_metric = Metric.TEST_CLEAN_ACC.value
plot_reg(x_metric, y_metric)
# %%
x_metric = Metric.TEST_CLEAN_ACC.value
for metric in metric_ba_test_list:
    y_metric = metric.value
    if y_metric == x_metric:
        continue
    print(f"{x_metric} vs. {y_metric}")
    plot_reg(x_metric, y_metric)
# %%
x_metric = Metric.TEST_CLEAN_ACC.value
for metric in metric_aa_list:
    y_metric = metric.value
    print(f"{x_metric} vs. {y_metric}")
    plot_reg(x_metric, y_metric)
# %%
df_aa = df[df["test_aa_acc"].notna()]
cols = metric_val_list+metric_ba_test_list+metric_aa_list
cols = [m.value for m in cols]
df_aa = df_aa[cols]
df_aa
# %%
fig = plt.figure(figsize=(10, 10), dpi=300)
corr_m = df_aa.corr(method="kendall")
sns.heatmap(corr_m, annot=True, square=True)
plt.show()
# %%
# ============= Best Archs =============

# %%
df_aa = df[df["test_aa_acc"].notna()]
df_best_aa = df_aa.sort_values(by="test_aa_acc", ascending=False)[:20]
df_best_aa

# %%

fig = plt.figure(figsize=(15, 5), dpi=300)
plt.subplot(1, 6, 1)
x_d_axis = dataset.search_space.depth1
x_w_axis = dataset.search_space.width1
colors = sns.color_palette("hls", n_colors=20)

xlabels = ["Depth1", "Width1", "Depth2", "Width2", "Depth3", "Width3"]

xss = np.zeros((20, 6), dtype=np.int32)
yss = np.zeros((20, 6), dtype=np.float32)

for i, (_, row) in enumerate(df_best_aa.iterrows()):
    print(f"{row['arch_id']} {row['test_aa_acc']}")
    arch_tuple = row["arch"]
    xss[i] = np.array(arch_tuple, dtype=np.int32)
    yss[i] = row["test_aa_acc"]


for i in range(6):
    plt.subplot(1, 6, i+1)
    for j, c, arch_id in zip(range(20), colors, df_best_aa["arch_id"]):
        plt.scatter(xss[j, i], yss[j, i], c=c, s=10, label=arch_id)

    if i%2 == 0:
        plt.xticks(x_d_axis)
        plt.xlim(3,12)
    else:
        plt.xticks(x_w_axis)
        plt.xlim(7,17)

    plt.xlabel(xlabels[i])
    if i == 5:
        plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    # plt.tight_layout()

fig.tight_layout()
plt.show()

#%%
def plot_hist(x, xlabel, y_metric, df_new=None, **kwargs):
    if df_new is None:
        df_new = df
    df_temp=df_new[df_new[y_metric].notna()]
    fig = plt.figure(figsize=(10, 5), dpi=300)

    default_opts = dict(
        alpha=0.5, cbar=True, palette="flare",
        stat="probability"
    )
    default_opts.update(kwargs)
    
    sns.histplot(x=x, y=df_temp[y_metric], **default_opts)
    plt.xlabel(xlabel)
    plt.show()

# %%
def plot_violin(x, xlabel, y_metric, df_new=None, **kwargs):
    if df_new is None:
        df_new = df
    df_temp=df_new[df_new[y_metric].notna()]
    fig = plt.figure(figsize=(10, 5), dpi=300)

    default_opts = dict(
        orient="x",
        alpha=0.5,
        palette="flare", hue=x,
        density_norm="count",
    )
    default_opts.update(kwargs)
    
    # sns.histplot(x=x, y=df_temp[y_metric], **default_opts)
    sns.violinplot(x=x, y=df_temp[y_metric], **default_opts)
    plt.xlabel(xlabel)
    plt.legend(loc="lower right")
    plt.show()
# %%
df_ba=df[df["test_clean_acc"].notna()]
for i in range(6):
    x=[arch_tuple[i] for arch_tuple in df_ba["arch"]]
    for metric in metric_ba_test_list:
        y_metric = metric.value
        print(f"{x_metric} vs. {y_metric}")
        # plot_hist(x, xlabels[i], y_metric, df_new=df_ba)
        plot_violin(x, xlabels[i], y_metric, df_new=df_ba)


# %%
df_aa=df[df["test_aa_acc"].notna()]
for i in range(6):
    x=[arch_tuple[i] for arch_tuple in df_aa["arch"]]
    y_metric = Metric.TEST_AA_ACC.value
    # plot_hist(x, xlabels[i], y_metric, df_new=df_aa)
    plot_violin(x, xlabels[i], y_metric, df_new=df_aa)
# %%
