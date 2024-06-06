# NAS-Bench-R

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