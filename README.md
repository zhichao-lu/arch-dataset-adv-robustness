# NAS-Bench-R

## Instruction

Download and put `cifar10.jsonl` to path `data/`.

## Benchmark

```bash
python nas_benchmark.py -m algo=random_search,local_search,regularized_evolution seed="range(200)"

python nas_benchmark.py -m algo=bananas seed="range(200)"
```

# Acknowledgement

* [naszilla](https://github.com/naszilla/naszilla)