import os
from enum import Enum
from dataclasses import dataclass, field, fields
import orjsonl
# from glom import glom
import numpy as np
from typing import Any, List, Union, Iterator
import datetime

from search_space import Arch, WRNSearchSpace


class Metric(Enum):
    """
        Scalar Metrics for objective
    """
    MACS = "macs"
    PARAMS = "params"

    # TRAIN_ACC='train_acc'
    # TRAIN_LOSS='train_loss'
    # VAL_ACC='val_acc'
    # VAL_LOSS='val_loss'
    BEST_EPOCH = 'best_epoch'
    VAL_CLEAN_ACC = "val_clean_acc"
    VAL_TEST_ACC = 'val_test_acc'
    VAL_PGD_ACC = "val_pgd_acc"
    VAL_CW_ACC = "val_cw_acc"

    # ====== only available at final test stage ======
    TEST_CLEAN_ACC = "test_clean_acc"
    TEST_LOSS = "test_loss"

    TEST_FGSM_ACC = "test_fgsm_acc"
    TEST_FGSM_STABLE = "test_fgsm_stable"

    TEST_PGD_ACC = "test_pgd_acc"
    TEST_PGD_STABLE = "test_pgd_stable"
    TEST_PGD_LIP = "test_pgd_lip"

    TEST_CW_ACC = "test_cw_acc"
    TEST_CW_STABLE = "test_cw_stable"
    TEST_CW_LIP = "test_cw_lip"

    TEST_AA_ACC = "test_aa_acc"

    TEST_CORRUPT_ACC = "corrupt_acc"


@dataclass
class TrainHistory:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_clean_acc: List[float]
    val_pgd_acc: List[float]
    val_pgd_stable: List[float]
    val_pgd_lip: List[float]
    val_cw_acc: List[float]
    val_cw_stable: List[float]
    val_cw_lip: List[float]



@dataclass
class CorruptAcc:
    gaussian_noise: float
    shot_noise: float
    impulse_noise: float
    defocus_blur: float
    glass_blur: float
    motion_blur: float
    zoom_blur: float
    snow: float
    frost: float
    fog: float
    brightness: float
    contrast: float
    elastic_transform: float
    pixelate: float
    jpeg_compression: float
    total_avg: float


@dataclass
class AttackAcc:
    Linf_fgsm_acc: float = field(default=None)
    Linf_fgsm_stable: float = field(default=None)
    Linf_pgd_acc: float = field(default=None)
    Linf_pgd_stable: float = field(default=None)
    Linf_pgd_lip: float = field(default=None)
    Linf_cw_acc: float = field(default=None)
    Linf_cw_stable: float = field(default=None)
    Linf_cw_lip: float = field(default=None)
    Linf_aa_acc: float = field(default=None)  # AutoAttack

    # L2_pgd_acc: float = field(default=None)
    # L2_pgd_stable: float = field(default=None)
    # L2_pgd_stable: float = field(default=None)
    # L2_cw_acc: float = field(default=None)
    # L2_aa_acc: float = field(default=None)  # AutoAttack

    # corruption: CorruptAcc = field(default=None)


@dataclass
class TrainRecord:
    """
    One training Record instance for a model
    """
    seed: int
    gpu_type: str

    timestamp: datetime.date

    history: TrainHistory
    # val_pgd_acc: float
    # val_cw_acc: float

    best_epoch: int

    test_clean_acc: float = field(default=None)
    test_loss: float = field(default=None)

    attack_acc: AttackAcc = field(default=None)
    corruption_acc: CorruptAcc = field(default=None)
    # last_model_attack_acc: AttackAcc = field(default=None)  # not supported yet


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
            Metric.TEST_LOSS,
            Metric.TEST_FGSM_ACC,
            Metric.TEST_PGD_ACC,
            Metric.TEST_PGD_STABLE,
            Metric.TEST_PGD_LIP,
            Metric.TEST_CW_ACC,
            Metric.TEST_CW_STABLE,
            Metric.TEST_CW_LIP,
            Metric.TEST_AA_ACC,
            Metric.TEST_CORRUPT_ACC
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

                corrupt_acc_dict = _test_best_record.get("corruptions", {})
                if len(corrupt_acc_dict)>0:
                    # corrupt_acc_dict["total_avg"] = dict(
                    #     loss=np.mean([v["loss"] for v in _test_best_record["corruption"].values()]),
                    #     acc=np.mean([v["acc"] for v in _test_best_record["corruption"].values()])
                    # )
                    corrupt_acc_dict["total_avg"] = np.mean([v["acc"] for v in _test_best_record["corruptions"].values()])
                    corruption = CorruptAcc(**corrupt_acc_dict)
                else:
                    corruption = None

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
                            val_pgd_stable=_train_record["pgd_stable"],
                            val_pgd_lip=_train_record["pgd_lip"],
                            val_cw_acc=_train_record["cw"],
                            val_cw_stable=_train_record["cw_stable"],
                            val_cw_lip=_train_record["cw_lip"],
                        ),
                        # val_pgd_acc=_train_record["best_pgd"],
                        # val_cw_acc=_train_record["best_cw"],
                        # val_pgd_acc=_train_record["pgd"][_train_record["best_epoch"]],
                        # val_cw_acc=_train_record["cw"][_train_record["best_epoch"]],

                        test_clean_acc=_test_best_record.get(
                            "test_clean_acc", None),
                        test_loss=_test_best_record.get("test_loss", None),

                        attack_acc=AttackAcc(
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
                            Linf_cw_lip=_test_best_record.get(
                                "Linf_cw_lip", None),
                            Linf_aa_acc=_test_best_record.get(
                                "Linf_aa_acc", None)
                        ),
                        corruption_acc=corruption
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
        elif metric == Metric.VAL_CLEAN_ACC:
            # Note: use val set acc
            best_model_val_clean_acc = [
                train_record.history.val_clean_acc[train_record.best_epoch]
                for train_record in record.train_records
            ]
            return np.mean(best_model_val_clean_acc)
        elif metric == Metric.VAL_PGD_ACC:
            best_model_val_pgd_acc = [
                train_record.history.val_pgd_acc[train_record.best_epoch]
                for train_record in record.train_records
            ]
            return np.mean(best_model_val_pgd_acc)
        elif metric == Metric.VAL_CW_ACC:
            best_model_val_cw_acc = [
                train_record.history.val_cw_acc[train_record.best_epoch]
                for train_record in record.train_records
            ]
            return np.mean(best_model_val_cw_acc)
        elif metric == Metric.TEST_CLEAN_ACC:
            test_clean_acc = [
                train_record.test_clean_acc
                for train_record in record.train_records
                if train_record.test_clean_acc is not None #TODO: remove these when all records are completed.
            ]
            return np.mean(test_clean_acc)
        elif metric == Metric.TEST_LOSS:
            test_loss = [
                train_record.test_loss
                for train_record in record.train_records
                if train_record.test_loss is not None
            ]
            return np.mean(test_loss)
        elif metric == Metric.TEST_FGSM_ACC:
            test_fgsm_acc = [
                train_record.attack_acc.Linf_fgsm_acc
                for train_record in record.train_records
                if train_record.attack_acc.Linf_fgsm_acc is not None
            ]
            return np.mean(test_fgsm_acc)
        elif metric == Metric.TEST_FGSM_STABLE:
            test_fgsm_stable = [
                train_record.attack_acc.Linf_fgsm_stable
                for train_record in record.train_records
                if train_record.attack_acc.Linf_fgsm_stable is not None
            ]
            return np.mean(test_fgsm_stable)
        elif metric == Metric.TEST_PGD_ACC:
            test_pgd_acc = [
                train_record.attack_acc.Linf_pgd_acc
                for train_record in record.train_records
                # if train_record.attack_acc.Linf_pgd_acc is not None
            ]
            return np.mean(test_pgd_acc)
        elif metric == Metric.TEST_PGD_STABLE:
            test_pgd_stable = [
                train_record.attack_acc.Linf_pgd_stable
                for train_record in record.train_records
                # if train_record.attack_acc.Linf_pgd_stable is not None
            ]
            return np.mean(test_pgd_stable)
        elif metric == Metric.TEST_PGD_LIP:
            test_pgd_lip = [
                train_record.attack_acc.Linf_pgd_lip
                for train_record in record.train_records
                if train_record.attack_acc.Linf_pgd_lip is not None
            ]
            return np.mean(test_pgd_lip)
        elif metric == Metric.TEST_CW_ACC:
            test_cw_acc = [
                train_record.attack_acc.Linf_cw_acc
                for train_record in record.train_records
                if train_record.attack_acc.Linf_cw_acc is not None
            ]
            return np.mean(test_cw_acc)
        elif metric == Metric.TEST_CW_STABLE:
            test_cw_stable = [
                train_record.attack_acc.Linf_cw_stable
                for train_record in record.train_records
                if train_record.attack_acc.Linf_cw_stable is not None
            ]
            return np.mean(test_cw_stable)
        elif metric == Metric.TEST_CW_LIP:
            test_cw_lip = [
                train_record.attack_acc.Linf_cw_lip
                for train_record in record.train_records
                if train_record.attack_acc.Linf_cw_lip is not None
            ]
            return np.mean(test_cw_lip)
        elif metric == Metric.TEST_AA_ACC:
            test_aa_acc = [
                train_record.attack_acc.Linf_aa_acc
                for train_record in record.train_records
                if train_record.attack_acc.Linf_aa_acc is not None
            ]
            return np.mean(test_aa_acc)
        elif metric == Metric.TEST_CORRUPT_ACC:
            test_corruption = [
                train_record.attack_acc.corruption.total_avg
                for train_record in record.train_records
                if train_record.attack_acc.corruption is not None
            ]
            return np.mean(test_corruption)
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
