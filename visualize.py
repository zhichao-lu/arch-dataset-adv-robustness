# %%
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from dataset import NASBenchR_CIFAR10_Dataset, Metric
from search_space import Arch
dataset = NASBenchR_CIFAR10_Dataset("data/cifar10_temp.jsonl")

root_path = Path("./fig")

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
metric_val_list = [Metric.VAL_BEST_CLEAN_ACC,
                   Metric.VAL_BEST_PGD_ACC,
                   Metric.VAL_BEST_CW_ACC]

metric_ba_test_list = [
    Metric.TEST_CLEAN_ACC,
    Metric.TEST_FGSM_ACC,
    Metric.TEST_PGD_ACC,
    Metric.TEST_CW_ACC,
]

# %% Create data to csv
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

    assert len(
        record.train_records) == 1, f"arch_{record.arch_id}: {len(record.train_records)}"
    for train_log in record.train_records:
        best_epoch = train_log.best_epoch
        history = train_log.history
        rec_dict.update(dict(
            best_epoch=best_epoch,
            train_last_acc=history.train_acc[-1],
            train_last_loss=history.train_loss[-1],
            train_best_model_acc=history.train_acc[best_epoch],
            train_best_model_loss=history.train_loss[best_epoch],
            val_best_model_acc=history.val_clean_acc[best_epoch],
            val_best_model_loss=history.val_loss[best_epoch],
            val_best_model_pgd=history.val_pgd_acc[best_epoch]
        ))

    dict_list.append(rec_dict)

df = pd.DataFrame.from_dict(dict_list)
df
# %%
df.to_csv("data/cifar10.csv", index=False)

# %% load csv to data
df = pd.read_csv("data/cifar10.csv")
df["arch"] = pd.Series([eval(arch) for arch in df["arch"]])
df.sort_values(by="arch_id", inplace=True)
df.reset_index(drop=True, inplace=True)
df
# %%
# ============= Plotting =============
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
def plot_hist(x_metric, df_temp, xlabel=None, save=True, suffix="", **kwargs):
    fig = plt.figure(figsize=(10, 3), dpi=300)
    default_opts = dict(
        alpha=0.5,
        ec="None",
        stat="probability", kde=True,
        bins=250,
    )
    default_opts.update(kwargs)

    sns.histplot(x=x_metric, data=df_temp, **default_opts)
    if xlabel is not None:
        plt.xlabel(xlabel)

    if suffix != "":
        suffix = "_" + suffix

    if save:
        plt.savefig(root_path/f"{x_metric if xlabel is None else xlabel}_hist{suffix}.png",
                    bbox_inches='tight')
    plt.show()


# %%
df_full = pd.read_csv("debug/RobustNASBench_v1.csv")
plot_hist("Params", df_full, bins=250,
          xlabel=Metric.PARAMS.value, suffix="full")
plot_hist("FLOPs", df_full, bins=250, xlabel=Metric.MACS.value, suffix="full")
# %%
plot_hist(Metric.PARAMS.value, df)
plot_hist(Metric.MACS.value, df)

# %%
df_temp = df[df[Metric.TEST_CLEAN_ACC.value].notna()]
plot_hist(Metric.PARAMS.value, df_temp, suffix="ba")
plot_hist(Metric.MACS.value, df_temp, suffix="ba")

# %%
df_temp = df[df[Metric.TEST_AA_ACC.value].notna()]
plot_hist(Metric.PARAMS.value, df_temp)
plot_hist(Metric.MACS.value, df_temp)

# %%


def plot_scatter(x_metric, y_metric, df_new=None, ylim=None, save=True, suffix="", **kwargs):
    if df_new is None:
        df_new = df
    df_temp = df_new[df_new[y_metric].notna()]

    if ylim is not None:
        df_temp = df_temp[df_temp[y_metric] > ylim[0]]
        df_temp = df_temp[df_temp[y_metric] < ylim[1]]

    fig = plt.figure(figsize=(10, 10), dpi=300)
    default_opts = dict(
        alpha=0.5, s=25, legend=False, ec=None
    )
    default_opts.update(kwargs)

    sns.scatterplot(data=df_temp, x=x_metric, y=y_metric,
                    hue=y_metric, **default_opts)
    # if ylim is not None:
    #     plt.ylim(*ylim)

    if suffix != "":
        suffix = "_" + suffix

    if save:
        plt.savefig(
            root_path /
            f"{x_metric.replace('_', '-')}_{y_metric.replace('_', '-')}_scatter{suffix}.png",
            bbox_inches='tight')

    plt.show()


# %%
fig = plt.figure(figsize=(10, 10), dpi=300)

df1 = df[df["test_clean_acc"].notna()]
df1 = df1[df1["test_clean_acc"] > 84]

sns.scatterplot(data=df1, x="params", y="test_clean_acc",
                hue="test_clean_acc", alpha=0.5, s=25, legend=False, ec=None)
# plt.savefig(root_path/"params_test_clean_acc_scatter.pdf")
plt.show()


# %%
x_metric = Metric.PARAMS.value
for metric in metric_val_list:
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)

# %%
x_metric = Metric.MACS.value
for metric in metric_val_list:
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)

# %%
x_metric = Metric.PARAMS.value
for metric, ylim in zip(metric_ba_test_list, [(85, 89), (58, 63), (52, 58), (50, 56)]):
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric, ylim=ylim)
# %%
x_metric = Metric.MACS.value
for metric, ylim in zip(metric_ba_test_list, [(85, 89), (58, 63), (52, 58), (50, 56)]):
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric, ylim=ylim)
# %%
x_metric = Metric.PARAMS.value
for metric in metric_aa_list:
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)
# %%
x_metric = Metric.MACS.value
for metric in metric_aa_list:
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_scatter(x_metric, y_metric)
# %%
x_metric = Metric.VAL_BEST_CLEAN_ACC.value
y_metric = Metric.TEST_CLEAN_ACC.value
plot_scatter(x_metric, y_metric, ylim=(84, 89))
# %%


def plot_reg(x_metric, y_metric, df_new=None, ylim=None, save=True, suffix="", **kwargs):
    if df_new is None:
        df_new = df
    df_temp = df_new[df_new[y_metric].notna()]

    if ylim is not None:
        df_temp = df_temp[df_temp[y_metric] > ylim[0]]
        df_temp = df_temp[df_temp[y_metric] < ylim[1]]

    fig = plt.figure(figsize=(10, 10), dpi=300)
    default_opts = dict(
        scatter_kws=dict(alpha=0.5, s=2, color='black'),
        line_kws=dict(color="red", alpha=0.3),
        ci=95
    )
    default_opts.update(kwargs)

    sns.regplot(data=df_temp, x=x_metric, y=y_metric,  **default_opts)
    slope, intercept, r, p, se = scipy.stats.linregress(
        df_temp[x_metric], df_temp[y_metric])
    # slope, intercept = np.polyfit(df_new[x_metric], df_new[y_metric], 1)
    # _x = [df_temp[x_metric].min(), df_temp[x_metric].max()]
    # _y = [slope*x+intercept for x in _x]
    # plt.plot(_x, _y, color="green", alpha=0.3)
    tau = scipy.stats.kendalltau(df_temp[x_metric], df_temp[y_metric])

    plt.text(0.05, 0.95, f'y = {slope:.4f}x + {intercept:.4f}, r = {r:.4f}\nτ = {tau.statistic:.4f}',
             ha='left', va='top', transform=plt.gca().transAxes)
    print(f"slope: {slope:.4f} ± {se:.4f}")

    if suffix != "":
        suffix = "_" + suffix

    if save:
        plt.savefig(
            root_path /
            f"{x_metric.replace('_', '-')}_{y_metric.replace('_', '-')}_regress{suffix}.png",
            bbox_inches='tight')

    plt.show()


# %%
x_metric = Metric.VAL_BEST_CLEAN_ACC.value
y_metric = Metric.TEST_CLEAN_ACC.value
plot_reg(x_metric, y_metric, ylim=(84, 89))
# %%
x_metric = Metric.TEST_CLEAN_ACC.value
for metric, ylim in zip(metric_ba_test_list, [None, (56, 64), (50, 58), (50, 58)]):
    # for metric in metric_ba_test_list:
    y_metric = metric
    if y_metric == x_metric:
        continue
    print(f"{x_metric} vs. {y_metric}")
    plot_reg(x_metric, y_metric, ylim=ylim)
# %%
x_metric = Metric.TEST_CLEAN_ACC.value
for metric in metric_ba_test_list:
    y_metric = metric
    if y_metric == x_metric:
        continue
    print(f"{x_metric} vs. {y_metric}")
    plot_reg(x_metric, y_metric, suffix="raw")
# %%
x_metric = Metric.VAL_BEST_CLEAN_ACC.value
for metric, ylim in zip(metric_ba_test_list, [(84, 89), (56, 64), (50, 58), (50, 58)]):
    # for metric in metric_ba_test_list:
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_reg(x_metric, y_metric, ylim=ylim)
# %%
x_metric = Metric.VAL_BEST_CLEAN_ACC.value
for metric in metric_ba_test_list:
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_reg(x_metric, y_metric, suffix="raw")
# %%
x_metric = Metric.TEST_CLEAN_ACC.value
for metric in metric_aa_list:
    y_metric = metric
    print(f"{x_metric} vs. {y_metric}")
    plot_reg(x_metric, y_metric)
# %%
# ========== Correlation =============
# %%
df_ba = df[df["test_clean_acc"].notna()]
cols = metric_val_list+metric_list2 + metric_ba_test_list
df_ba = df_ba[cols]
df_ba
# %%
fig = plt.figure(figsize=(15, 15), dpi=300)
corr_m = df_ba.corr(method="kendall")
sns.heatmap(corr_m, annot=True, square=True, fmt=".2g")
plt.savefig(root_path/"corr_ba.png", bbox_inches='tight')
plt.show()
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
topk = 20
# df_aa = df[df["test_aa_acc"].notna()]
# df_best_aa = df_aa.sort_values(by="test_aa_acc", ascending=False)[:topk]
# df_best_aa

df_pgd = df[df[Metric.TEST_PGD_ACC.value].notna()]
df_best_pgd = df_pgd.sort_values(
    by=Metric.TEST_PGD_ACC.value, ascending=False)[:topk]
df_best_pgd

# %%
df_best_arch = df_best_pgd
fig = plt.figure(figsize=(15, 5), dpi=300)
plt.subplot(1, 6, 1)
x_d_axis = dataset.search_space.depth1
x_w_axis = dataset.search_space.width1
colors = sns.color_palette("hls", n_colors=topk)

xlabels = ["Depth1", "Depth2", "Depth3", "Width1", "Width2", "Width3"]

xss = np.zeros((topk, 6), dtype=np.int32)
yss = np.zeros((topk, 6), dtype=np.float32)

for i, (_, row) in enumerate(df_best_arch.iterrows()):
    print(f"{row['arch_id']} {row[Metric.TEST_PGD_ACC.value]}")
    arch_tuple = row["arch"]
    xss[i] = np.array(arch_tuple, dtype=np.int32)
    yss[i] = row[Metric.TEST_PGD_ACC.value]

xss = np.concatenate([xss[:, ::2], xss[:, 1::2]], axis=1)

for i in range(6):
    plt.subplot(1, 6, i+1)
    for j, c, arch_id in zip(range(topk), colors, df_best_arch["arch_id"]):
        plt.scatter(xss[j, i], yss[j, i], c=c, s=10, label=arch_id)

    if i < 3:
        plt.xticks(x_d_axis)
        plt.xlim(3, 12)
    else:
        plt.xticks(x_w_axis)
        plt.xlim(7, 17)

    plt.xlabel(xlabels[i])
    if i == 5:
        plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    # plt.tight_layout()

fig.tight_layout()
plt.savefig(root_path/"best-top20-arch.png", bbox_inches='tight')
plt.show()


# %%
xlabels = ["Depth1", "Width1", "Depth2", "Width2", "Depth3", "Width3"]
# %%


def plot_violin(x, xlabel, y_metric, df_new=None, topk=50, save=True, suffix="", **kwargs):
    if df_new is None:
        df_new = df
    df_temp = df_new[df_new[y_metric].notna()]
    topk_args = df_temp[y_metric].argsort()[-topk:]

    _x = x[topk_args]
    _y = df_temp[y_metric].sort_values()[-topk:].values

    fig = plt.figure(figsize=(10, 5), dpi=300)

    default_opts = dict(
        orient="x",
        alpha=0.5,
        palette="flare", hue=_x,
        legend=False,
        density_norm="count",
    )
    default_opts.update(kwargs)
    assert len(_x) == len(_y)
    # sns.histplot(x=x, y=df_temp[y_metric], **default_opts)
    sns.violinplot(x=_x, y=_y, **default_opts)
    plt.xlabel(xlabel)
    plt.ylabel(y_metric)
    plt.grid(axis="y")
    # plt.legend(loc="lower right")

    if suffix != "":
        suffix = "_" + suffix

    if save:
        plt.savefig(
            root_path /
            f"best-top{topk}-arch_{xlabel}_{y_metric.replace('_', '-')}{suffix}.png",
            bbox_inches='tight')

    plt.show()


# %%
def plot_stripplot(x, xlabel, y_metric, df_new=None, topk=100, save=True, suffix="", **kwargs):
    if df_new is None:
        df_new = df
    df_temp = df_new[df_new[y_metric].notna()]
    topk_args = df_temp[y_metric].argsort()[-topk:]

    _x = x[topk_args]
    _y = df_temp[y_metric].sort_values()[-topk:].values

    fig = plt.figure(figsize=(10, 5), dpi=300)

    default_opts = dict(
        orient="x",
        alpha=0.6,
        palette="flare", hue=_x,
        legend=False,
        # jitter=0.2,
    )
    violin_default_opts = dict(
        orient="x",
        palette="flare", hue=_x,
        legend=False,
        density_norm="count",
    )

    default_opts.update(kwargs)
    assert len(_x) == len(_y)
    ax = sns.violinplot(x=_x, y=_y, **violin_default_opts)
    plt.setp(ax.collections, alpha=.2)
    sns.stripplot(x=_x, y=_y, ax=ax, **default_opts)
    plt.xlabel(xlabel)
    plt.ylabel(y_metric)
    plt.grid(axis="y")
    # plt.legend(loc="lower right")

    if suffix != "":
        suffix = "_" + suffix

    if save:
        plt.savefig(
            root_path /
            f"best-top{topk}-arch_{xlabel}_{y_metric.replace('_', '-')}{suffix}.png",
            bbox_inches='tight')

    plt.show()

# %%


df_ba = df[df["test_clean_acc"].notna()]
for i in range(6):
    x = np.array([arch_tuple[i] for arch_tuple in df_ba["arch"]])
    for metric in metric_ba_test_list:
        y_metric = metric.value
        print(f"{xlabels[i]} vs. {y_metric}")
        # plot_hist(x, xlabels[i], y_metric, df_new=df_ba)
        # plot_violin(x, xlabels[i], y_metric, df_new=df_ba)
        plot_stripplot(x, xlabels[i], y_metric, df_new=df_ba, topk=100)


# %%
df_aa = df[df["test_aa_acc"].notna()]
for i in range(6):
    x = [arch_tuple[i] for arch_tuple in df_aa["arch"]]
    y_metric = Metric.TEST_AA_ACC.value
    # plot_hist(x, xlabels[i], y_metric, df_new=df_aa)
    plot_violin(x, xlabels[i], y_metric, df_new=df_aa)
# %%
# %%


def plot_hist(x, xlabel, y_metric, x_ticks, df_new=None, topk=100, save=True, suffix="", **kwargs):
    if df_new is None:
        df_new = df
    df_temp = df_new[df_new[y_metric].notna()]

    topk_args = df_temp[y_metric].argsort()[-topk:]

    _x = x[topk_args]

    fig = plt.figure(figsize=(10, 5), dpi=300)

    bins = len(x_ticks)

    default_opts = dict(
        alpha=0.5,
        discrete=True,
        bins=bins,
        binrange=(0, bins),
        # ec=None,
        shrink=0.8,
        stat="probability"
    )
    default_opts.update(kwargs)

    ax = sns.histplot(x=_x, **default_opts)
    plt.xlabel(f"{xlabel} vs. {y_metric}")
    plt.xlim(-0.5, bins-0.5)

    if x_ticks is not None:
        x_ticks = [str(t) for t in x_ticks]
        plt.xticks(ticks=list(range(len(x_ticks))), labels=x_ticks)

    if suffix != "":
        suffix = "_" + suffix

    if save:
        plt.savefig(
            root_path /
            f"best-top{topk}-arch_{xlabel}_{y_metric.replace('_', '-')}_dist{suffix}.png",
            bbox_inches='tight')

    plt.show()


# %%
df_ba = df[df["test_clean_acc"].notna()]
x_d_axis = dataset.search_space.depth1
x_w_axis = dataset.search_space.width1

depth_map = {d: i for i, d in enumerate(x_d_axis)}
width_map = {d: i for i, d in enumerate(x_w_axis)}

for i in range(6):
    if i % 2 == 0:
        x = np.array([
            depth_map[arch_tuple[i]]
            for arch_tuple in df_ba["arch"]
        ])

        for metric in metric_ba_test_list:
            y_metric = metric.value
            print(f"{xlabels[i]} vs. {y_metric}")

            plot_hist(x, xlabels[i], y_metric, x_ticks=x_d_axis, df_new=df_ba)
    else:
        x = np.array([
            width_map[arch_tuple[i]]
            for arch_tuple in df_ba["arch"]
        ])

        for metric in metric_ba_test_list:
            y_metric = metric.value
            print(f"{xlabels[i]} vs. {y_metric}")

            plot_hist(x, xlabels[i], y_metric, x_ticks=x_w_axis, df_new=df_ba)

# %%
