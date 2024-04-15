import os
from enum import Enum
from dataclasses import dataclass, field
import orjsonl
# from glom import glom
import numpy as np
from typing import Any, List, Union, Iterator
import datetime

from search_space import Arch, WRNSearchSpace


class Metric(Enum):
    """
        Metrics for objective
    """
    MACS = "macs"
    PARAMS = "params"

    # TRAIN_ACC='train_acc'
    # TRAIN_LOSS='train_loss'
    # VAL_ACC='val_acc'
    # VAL_LOSS='val_loss'
    BEST_EPOCH = 'best_epoch'
    VAL_BEST_CLEAN_ACC = "val_best_clean_acc"
    VAL_BEST_PGD_ACC = "val_best_pgd_acc"
    VAL_BEST_CW_ACC = "val_best_cw_acc"

    # ====== only available at final test stage ======
    TEST_CORRUPT_ACC = "corrupt_acc"

    TEST_CLEAN_ACC = "test_clean_acc"
    TEST_FGSM_ACC = "test_fgsm_acc"
    TEST_FGSM_STABLE = "test_fgsm_stable"
    TEST_PGD_ACC = "test_pgd_acc"
    TEST_PGD_STABLE = "test_pgd_stable"
    TEST_PGD_LIP = "test_pgd_lip"
    TEST_CW_ACC = "test_cw_acc"
    TEST_CW_STABLE = "test_cw_stable"
    TEST_AA_ACC = "test_aa_acc"


@dataclass
class TrainHistory:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_clean_acc: List[float]
    val_pgd_acc: List[float]
    val_cw_acc: List[float]


@dataclass
class CorruptAcc:
    noise_acc: float
    # TODO: add more


@dataclass
class AttackAcc:
    Linf_fgsm_acc: float = field(default=None)
    Linf_fgsm_stable: float = field(default=None)
    Linf_pgd_acc: float = field(default=None)
    Linf_pgd_stable: float = field(default=None)
    Linf_pgd_lip: float = field(default=None)
    Linf_cw_acc: float = field(default=None)
    Linf_cw_stable: float = field(default=None)
    Linf_aa_acc: float = field(default=None)  # AutoAttack

    L2_pgd_acc: float = field(default=None)
    L2_pgd_stable: float = field(default=None)
    L2_pgd_stable: float = field(default=None)
    L2_cw_acc: float = field(default=None)
    L2_aa_acc: float = field(default=None)  # AutoAttack


@dataclass
class TrainRecord:
    """
    One training Record instance for a model
    """
    seed: int
    gpu_type: str

    timestamp: datetime.date

    history: TrainHistory
    val_best_pgd_acc: float
    val_best_cw_acc: float

    best_epoch: int

    test_clean_acc: float = field(default=None)
    test_corrupt_acc: CorruptAcc = field(default=None)  # not supported yet

    best_model_attack_acc: AttackAcc = field(default=None)
    last_model_attack_acc: AttackAcc = field(default=None)  # not supported yet


@dataclass
class Record:
    arch_id: int  # id (sorted by macs)
    arch: Arch
    macs: float
    params: float
    train_records: List[TrainRecord]


class NASBenchR_CIFAR10_Dataset:
    def __init__(self, data_path: str = './data/cifar10.jsonl'):
        raw_data = orjsonl.load(data_path)

        self.search_space = WRNSearchSpace()

        self.test_metrics = [
            Metric.TEST_CLEAN_ACC,
            Metric.TEST_FGSM_ACC,
            Metric.TEST_PGD_ACC,
            Metric.TEST_CW_ACC,
            Metric.TEST_AA_ACC,
        ]

        self.dataset = dict()  # Arch -> Record
        for raw_record in raw_data:
            _arch = raw_record["arch"]
            arch = Arch(_arch["D1"], _arch["W1"], _arch["D2"],
                        _arch["W2"], _arch["D3"], _arch["W3"])

            assert len(raw_record['train']) == 1 and len(
                raw_record['test_best']) == 1, f"arch_{raw_record['arch_id']}"

            train_records = []
            for _train_record, _test_best_record in zip(raw_record['train'], raw_record['test_best']):
                if _test_best_record is None:
                    _test_best_record = {}

                train_records.append(
                    TrainRecord(
                        seed=_train_record["seed"],
                        gpu_type=_train_record["gpu_type"],
                        timestamp=datetime.date(*[int(x) for x in _train_record["timestamp"].split("-")]),
                        best_epoch=_train_record["best_epoch"],
                        history=TrainHistory(
                            train_loss=_train_record["train_loss"],
                            train_acc=_train_record["train_acc"],
                            val_loss=_train_record["val_loss"],
                            val_clean_acc=_train_record["val_acc"],
                            val_pgd_acc=_train_record["pgd"],
                            val_cw_acc=_train_record["cw"],
                        ),
                        val_best_pgd_acc=_train_record["best_pgd"],
                        val_best_cw_acc=_train_record["best_cw"],

                        test_clean_acc=_test_best_record.get(
                            "clean_acc", None),
                        # test_corrupt_acc=CorruptAcc(
                        #     noise_acc=_train_record["test_noise_acc"],
                        # ),
                        best_model_attack_acc=AttackAcc(
                            Linf_fgsm_acc=_test_best_record.get(
                                "Linf_fgsm_acc", None),
                            Linf_fgsm_stable=_test_best_record.get(
                                "Linf_fgsm_stable", None),
                            Linf_pgd_acc=_test_best_record.get(
                                "Linf_pgd_acc", None),
                            Linf_pgd_stable=_test_best_record.get(
                                "Linf_pgd_stable", None),
                            Linf_pgd_lip=_test_best_record.get(
                                "Linf_pgd_lip", None),
                            Linf_cw_acc=_test_best_record.get(
                                "Linf_cw_acc", None),
                            Linf_cw_stable=_test_best_record.get(
                                "Linf_cw_stable", None),
                            Linf_aa_acc=_test_best_record.get(
                                "Linf_aa_acc", None),
                        ),
                    )
                )

            self.dataset[arch] = Record(
                arch_id=raw_record["arch_id"],
                arch=arch,
                macs=raw_record["MACs"],
                params=raw_record["Params"],
                train_records=train_records,
            )

    @property
    def name(self):
        return "NASBenchR_CIFAR10"

    def query(self, arch: Arch) -> Record:
        return self.dataset[arch]

    def batch_query(self, archs: List[Arch]) -> Iterator[Record]:
        for arch in archs:
            yield self.query(arch)

    def get_metric(self, record: Record, metric: Union[Metric, str]):
        if isinstance(metric, str):
            metric = Metric(metric)

        if metric == Metric.MACS:
            return record.macs
        elif metric == Metric.PARAMS:
            return record.params
        elif metric == Metric.VAL_BEST_CLEAN_ACC:
            # Note: use val set acc
            best_clean_acc = [
                np.max(train_record.history.val_clean_acc)
                for train_record in record.train_records
            ]
            return np.mean(best_clean_acc)
        elif metric == Metric.VAL_BEST_PGD_ACC:
            best_val_pgd_acc = [
                train_record.val_best_pgd_acc
                for train_record in record.train_records
            ]
            return np.mean(best_val_pgd_acc)
        elif metric == Metric.VAL_BEST_CW_ACC:
            best_val_pgd_acc = [
                train_record.val_best_cw_acc
                for train_record in record.train_records
            ]
            return np.mean(best_val_pgd_acc)
        elif metric == Metric.TEST_CLEAN_ACC:
            test_clean_acc = [
                train_record.test_clean_acc
                for train_record in record.train_records
            ]
            return np.mean(test_clean_acc)
        elif metric == Metric.TEST_CORRUPT_ACC:
            raise NotImplementedError()
        elif metric == Metric.TEST_FGSM_ACC:
            test_fgsm_acc = [
                train_record.best_model_attack_acc.Linf_fgsm_acc
                for train_record in record.train_records
            ]
            return np.mean(test_fgsm_acc)
        elif metric == Metric.TEST_FGSM_STABLE:
            test_fgsm_stable = [
                train_record.best_model_attack_acc.Linf_fgsm_stable
                for train_record in record.train_records
            ]
            return np.mean(test_fgsm_stable)
        elif metric == Metric.TEST_PGD_ACC:
            test_pgd_acc = [
                train_record.best_model_attack_acc.Linf_pgd_acc
                for train_record in record.train_records
            ]
            return np.mean(test_pgd_acc)
        elif metric == Metric.TEST_PGD_STABLE:
            test_pgd_stable = [
                train_record.best_model_attack_acc.Linf_pgd_stable
                for train_record in record.train_records
            ]
            return np.mean(test_pgd_stable)
        elif metric == Metric.TEST_PGD_LIP:
            test_pgd_lip = [
                train_record.best_model_attack_acc.Linf_pgd_lip
                for train_record in record.train_records
            ]
            return np.mean(test_pgd_lip)
        elif metric == Metric.TEST_CW_ACC:
            test_cw_acc = [
                train_record.best_model_attack_acc.Linf_cw_acc
                for train_record in record.train_records
            ]
            return np.mean(test_cw_acc)
        elif metric == Metric.TEST_CW_STABLE:
            test_cw_stable = [
                train_record.best_model_attack_acc.Linf_cw_stable
                for train_record in record.train_records
            ]
            return np.mean(test_cw_stable)
        elif metric == Metric.TEST_AA_ACC:
            test_aa_acc = [
                train_record.best_model_attack_acc.Linf_aa_acc
                for train_record in record.train_records
            ]
            return np.mean(test_aa_acc)
        else:
            raise ValueError(f"Invalid metric: {metric}")

    def get_objective(self, arch: Arch, metric: Union[Metric, str]):
        if isinstance(metric, str):
            metric = Metric(metric)
            
        record = self.query(arch)
        if metric in self.test_metrics:
            raise ValueError(
                f"Metric {metric} is only available at test stage")

        return self.get_metric(record, metric)
