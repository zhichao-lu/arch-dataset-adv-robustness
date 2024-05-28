import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import NASBenchR_CIFAR10_Dataset, Metric
from search_space import Arch
from optimizer.metabase import MetaOptimizer
import re

import random
import numpy as np
import torch
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

import logging
import json
import dataclasses
import datetime
import pickle


def get_output_dir():
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir).absolute()
    else:
        raise ValueError


def search(optimizer_cls, dataset, cfg):
    optimizer: MetaOptimizer = optimizer_cls(cfg, dataset)

    optimizer.run()

    # best_arch_list = optimizer.get_final_arch()
    top_arch_list = optimizer.get_topk_archs(5)
    records = list(dataset.batch_query(top_arch_list))
    return records


def print_record(record):
    logging.info(
        f"{record.arch_id}: "
        f"val clean acc: {dataset.get_metric(record, Metric.VAL_CLEAN_ACC):.2f} "
        f"val pgd acc: {dataset.get_metric(record, Metric.VAL_PGD_ACC):.2f} "
        f"val cw acc: {dataset.get_metric(record, Metric.VAL_CW_ACC):.2f} "
        f"test clean acc: {dataset.get_metric(record, Metric.TEST_CLEAN_ACC):.2f} "
        f"test fgsm acc: {dataset.get_metric(record, Metric.TEST_FGSM_ACC):.2f} "
        f"test pgd acc: {dataset.get_metric(record, Metric.TEST_PGD_ACC):.2f} "
        f"test cw acc: {dataset.get_metric(record, Metric.VAL_CW_ACC):.2f} "
    )


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return self._dataclass_to_dict(o)
        elif isinstance(o, datetime.date):
            return o.isoformat()

        return super().default(o)

    def _dataclass_to_dict(self, o):
        new_dict = {}
        for k, v in dataclasses.asdict(o).items():
            if dataclasses.is_dataclass(v):
                new_dict[k] = self._dataclass_to_dict(v)
            else:
                new_dict[k] = v
        return new_dict

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    OmegaConf.set_readonly(cfg, True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    optimizer_cls = hydra.utils.get_class(cfg.optimizer_cls)

    records = search(optimizer_cls, dataset, cfg)

    for record in records:
        print_record(record)

    output_path = get_output_dir()
    with (output_path/"results.json").open("w") as f:
        json.dump(records, f, cls=EnhancedJSONEncoder, indent=4)


if __name__ == "__main__":
    OmegaConf.register_new_resolver(
        "sanitize_dirname",
        lambda path: re.sub(r'/', '_', path)
    )
    dataset = NASBenchR_CIFAR10_Dataset("data/cifar10.jsonl")
    main()
