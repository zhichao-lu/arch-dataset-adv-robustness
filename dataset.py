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
    Note: This does not include all metrics, just the most common ones
    to ease the search procedure. If need more metrics, use `record`
    from `dataset.query()` directly.
    """

    MACS = "macs"
    PARAMS = "params"

    BEST_EPOCH = "best_epoch"

    # VAL metrics are evaluated based on the model at best epoch
    TRAIN_ACC = "train_acc"
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"
    VAL_CLEAN_ACC = "val_clean_acc"
    VAL_TEST_ACC = "val_test_acc"
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

    TEST_CORRUPT_ACC = "test_corrupt_acc"  # average of all corruptions
    # TEST_GAUSSIAN_NOISE = "test_corrupt_gaussian_noise"
    # TEST_SHOT_NOISE = "test_corrupt_shot_noise"
    # TEST_IMPULSE_NOISE = "test_corrupt_impulse_noise"
    # TEST_DEFOCUS_BLUR = "test_corrupt_defocus_blur"
    # TEST_GLASS_BLUR = "test_corrupt_glass_blur"
    # TEST_MOTION_BLUR = "test_corrupt_motion_blur"
    # TEST_ZOOM_BLUR = "test_corrupt_zoom_blur"
    # TEST_SNOW = "test_corrupt_snow"
    # TEST_FROST = "test_corrupt_frost"
    # TEST_FOG = "test_corrupt_fog"
    # TEST_BRIGHTNESS = "test_corrupt_brightness"
    # TEST_CONTRAST = "test_corrupt_contrast"
    # TEST_ELASTIC_TRANSFORM = "test_corrupt_elastic_transform"
    # TEST_PIXELATE = "test_corrupt_pixelate"
    # TEST_JPEG_COMPRESSION = "test_corrupt_jpeg_compression"


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
class CorruptionEntry:
    acc: float
    loss: float


@dataclass
class CorruptionData:
    # TDDO: add more corruption types with 5 levels
    gaussian_noise: CorruptionEntry
    shot_noise: CorruptionEntry
    impulse_noise: CorruptionEntry
    defocus_blur: CorruptionEntry
    glass_blur: CorruptionEntry
    motion_blur: CorruptionEntry
    zoom_blur: CorruptionEntry
    snow: CorruptionEntry
    frost: CorruptionEntry
    fog: CorruptionEntry
    brightness: CorruptionEntry
    contrast: CorruptionEntry
    elastic_transform: CorruptionEntry
    pixelate: CorruptionEntry
    jpeg_compression: CorruptionEntry
    total_avg: CorruptionEntry


@dataclass
class AttackData:
    Linf_fgsm_acc: float
    Linf_fgsm_stable: float
    Linf_pgd_acc: float
    Linf_pgd_stable: float
    Linf_pgd_lip: float
    Linf_cw_acc: float
    Linf_cw_stable: float
    Linf_cw_lip: float
    Linf_aa_acc: float  # AutoAttack

    # Not supported yet
    # L2_pgd_acc: float
    # L2_pgd_stable: float
    # L2_pgd_stable: float
    # L2_cw_acc: float
    # L2_aa_acc: float


@dataclass
class TrainRecord:
    """
    One training Record instance for a model
    """

    seed: int

    gpu_type: str  # debug info
    timestamp: datetime.date  # debug info
    training_time: float  # GPU days, debug info

    history: TrainHistory

    best_epoch: int
    test_clean_acc: float
    test_loss: float

    attack: List[AttackData]  # allow multiple attack runs due to randomness
    # one corruption acc for each model (no randomness)
    corruption: CorruptionData

    # Not supported yet
    # last_model_attack_acc: AttackAcc = field(default=None)


@dataclass
class Record:
    arch_id: int  # id (sorted by macs)
    arch: Arch
    macs: float
    params: float
    train_records: List[TrainRecord]


class NASBenchR_CIFAR10_Dataset:
    def __init__(self, data_path: str = "./data/cifar10.jsonl"):
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
            Metric.TEST_CORRUPT_ACC,
        ]

        self.dataset = dict()  # Arch -> Record
        for raw_record in raw_data:
            _arch = raw_record["arch"]
            arch = Arch(
                _arch["D1"],
                _arch["W1"],
                _arch["D2"],
                _arch["W2"],
                _arch["D3"],
                _arch["W3"],
            )

            assert (
                len(raw_record["train"]) == 1 and len(
                    raw_record["test_best"]) == 1
            ), f"arch_{raw_record['arch_id']}"

            train_records = []
            for _train_record, _test_best_record in zip(
                raw_record["train"], raw_record["test_best"]
            ):
                assert _test_best_record is not None

                _corruption_record = _test_best_record.get("corruptions", {})
                assert len(_corruption_record) > 0
                corrupt_acc_dict = {
                    k: CorruptionEntry(acc=v["acc"], loss=v["loss"])
                    for k,v in _corruption_record.items()
                }
                corrupt_acc_dict["total_avg"] = CorruptionEntry(
                    acc=np.mean([v["acc"] for v in _corruption_record.values()]),
                    loss=np.mean([v["loss"] for v in _corruption_record.values()]))
                
                corruption = CorruptionData(**corrupt_acc_dict)

                train_records.append(
                    TrainRecord(
                        seed=_train_record["seed"],
                        gpu_type=_train_record["gpu_type"],
                        timestamp=datetime.date(
                            *[int(x) for x in _train_record["timestamp"].split("-")]
                        ),
                        training_time=_train_record["training_time"],
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
                        test_clean_acc=_test_best_record.get(
                            "test_clean_acc", None),
                        test_loss=_test_best_record.get("test_loss", None),
                        attack=[
                            AttackData(
                                Linf_fgsm_acc=_test_best_record.get(
                                    "Linf_fgsm_acc", None
                                ),
                                Linf_fgsm_stable=_test_best_record.get(
                                    "Linf_fgsm_stable", None
                                ),
                                Linf_pgd_acc=_test_best_record.get(
                                    "Linf_pgd_acc", None
                                ),
                                Linf_pgd_stable=_test_best_record.get(
                                    "Linf_pgd_stable", None
                                ),
                                Linf_pgd_lip=_test_best_record.get(
                                    "Linf_pgd_lip", None
                                ),
                                Linf_cw_acc=_test_best_record.get(
                                    "Linf_cw_acc", None),
                                Linf_cw_stable=_test_best_record.get(
                                    "Linf_cw_stable", None
                                ),
                                Linf_cw_lip=_test_best_record.get(
                                    "Linf_cw_lip", None),
                                Linf_aa_acc=_test_best_record.get(
                                    "Linf_aa_acc", None),
                            )
                        ],
                        corruption=corruption,
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
        """
        Helper function to retrieve *scalar* metric from the record.
        """
        if isinstance(metric, str):
            metric = Metric(metric)

        if metric == Metric.MACS:
            return record.macs
        elif metric == Metric.PARAMS:
            return record.params
        elif metric == Metric.BEST_EPOCH:
            best_epochs = [
                train_record.best_epoch for train_record in record.train_records
            ]
            return np.mean(best_epochs)
        elif metric == Metric.TRAIN_ACC:
            train_acc = [
                train_record.history.train_acc[train_record.best_epoch]
                for train_record in record.train_records
            ]
            return np.mean(train_acc)
        elif metric == Metric.TRAIN_LOSS:
            train_loss = [
                train_record.history.train_loss[train_record.best_epoch]
                for train_record in record.train_records
            ]
            return np.mean(train_loss)
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
                train_record.test_clean_acc for train_record in record.train_records
            ]
            return np.mean(test_clean_acc)
        elif metric == Metric.TEST_LOSS:
            test_loss = [
                train_record.test_loss for train_record in record.train_records
            ]
            return np.mean(test_loss)
        elif metric == Metric.TEST_FGSM_ACC:
            test_fgsm_acc = [
                attack_acc.Linf_fgsm_acc
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_fgsm_acc)
        elif metric == Metric.TEST_FGSM_STABLE:
            test_fgsm_stable = [
                attack_acc.Linf_fgsm_stable
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_fgsm_stable)
        elif metric == Metric.TEST_PGD_ACC:
            test_pgd_acc = [
                attack_acc.Linf_pgd_acc
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_pgd_acc)
        elif metric == Metric.TEST_PGD_STABLE:
            test_pgd_stable = [
                attack_acc.Linf_pgd_stable
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_pgd_stable)
        elif metric == Metric.TEST_PGD_LIP:
            test_pgd_lip = [
                attack_acc.Linf_pgd_lip
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_pgd_lip)
        elif metric == Metric.TEST_CW_ACC:
            test_cw_acc = [
                attack_acc.Linf_cw_acc
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_cw_acc)
        elif metric == Metric.TEST_CW_STABLE:
            test_cw_stable = [
                attack_acc.Linf_cw_stable
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_cw_stable)
        elif metric == Metric.TEST_CW_LIP:
            test_cw_lip = [
                attack_acc.Linf_cw_lip
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_cw_lip)
        elif metric == Metric.TEST_AA_ACC:
            test_aa_acc = [
                attack_acc.Linf_aa_acc
                for train_record in record.train_records
                for attack_acc in train_record.attack
            ]
            return np.mean(test_aa_acc)
        elif metric == Metric.TEST_CORRUPT_ACC:
            test_corruption = [
                train_record.corruption.total_avg['acc']
                for train_record in record.train_records
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
