# %%
from dataset import NASBenchR_CIFAR10_Dataset,Metric
from search_space import Arch
dataset = NASBenchR_CIFAR10_Dataset("data/cifar10_temp.jsonl")
# %%


complete_archs = list(dataset.dataset.keys())
arch=complete_archs[0]
print(arch)

record = dataset.query(arch)

dataset.get_metric(record, Metric["VAL_BEST_CLEAN_ACC"])
# %%
record
# %%
