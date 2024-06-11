# 🐽 NARes: A Neural Architecture Dataset for Adversarial Robustness on WideResNet

## Search Space Overview

![](./nares_search_space.png "NARes Search Space")

Note:
* Decision Vector: $[D_1, W_1, D_2, W_2, D_3, W_3]$, where $D_{i\in\{1,2,3\}} \in \{4,5,7,9,11\}$ and $W_{i\in\{1,2,3\}} \in \{8,10,12,14,16\}$.
* Total Architectures: $5^6=15625$.
* Each architecture has an arch_id for identification, set from 1 to 15625 by the ascending order of \#MACs.

## Preparation

1. Install packages: `pip install -r requirements.txt`

2. Download and put `cifar10.jsonl` to path `data/`.

## Benchmark

```bash
# Support hydra syntax

# Run the NAS algorithms with seeds from 0 to 199
python nas_benchmark.py -m algo=random_search,local_search,regularized_evolution seed="range(200)"
python nas_benchmark.py -m algo=bananas seed="range(200)"
```

## Access the dataset

```python
from dataset import NASBenchR_CIFAR10_Dataset, Metric
from search_space import Arch

dataset = NASBenchR_CIFAR10_Dataset("data/cifar10.jsonl")

wrn_34_10 = Arch(
    depth1=5, depth2=5, depth3=5,
    width1=10, width2=10, width3=10,
)

wrn_70_16 = Arch(
    depth1=11, depth2=11, depth3=11,
    width1=16, width2=16, width3=16,
)

# get a single record
record = dataset.query(wrn_34_10)

# get multiple records at once (return a generator)
for record in dataset.batch_query([wrn_34_10, wrn_70_16]):
    ...
```


# Acknowledgement

* [naszilla](https://github.com/naszilla/naszilla)