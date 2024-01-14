import time
import json
import logging
import os
import copy
import torch
import numpy as np

from utils import AverageMeter

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Default implementation that handles dataloading and preparing batches, the
    train loop, gathering statistics, checkpointing and doing the final
    final evaluation.

    If this does not fulfil your needs free do subclass it and implement your
    required logic.
    """

    def __init__(self, optimizer, config):
        """
        Initializes the trainer.

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `get_config_from_args()`
        """
        self.optimizer = optimizer
        self.config = config
        self.epochs = self.config.search.epochs

        # preparations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # measuring stuff
        self.train_top1 = AverageMeter()
        self.train_top5 = AverageMeter()
        self.train_loss = AverageMeter()
        self.val_top1 = AverageMeter()
        self.val_top5 = AverageMeter()
        self.val_loss = AverageMeter()

        n_parameters = optimizer.get_model_size()
        # logger.info("param size = %fMB", n_parameters)
        self.search_trajectory = AttrDict(
            {
                "train_acc": [],
                "train_loss": [],
                "valid_acc": [],
                "valid_loss": [],
                "test_acc": [],
                "test_loss": [],
                "runtime": [],
                "train_time": [],
                "arch_eval": [],
                "params": n_parameters,
            }
        )

    def search(self, summary_writer=None, after_epoch: Callable[[int], None]=None, report_incumbent=True):
        """
        Start the architecture search.

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                train from scratch.
        """
        logger.info("Beginning search")

        np.random.seed(self.config.search.seed)
        torch.manual_seed(self.config.search.seed)

        self.optimizer.before_training()

        arch_weights = []
        for e in range(start_epoch, self.epochs):

            start_time = time.time()
            self.optimizer.new_epoch(e)


            end_time = time.time()
            # TODO: nasbench101 does not have train_loss, valid_loss, test_loss implemented, so this is a quick fix for now
            # train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss = self.optimizer.train_statistics()
            (
                train_acc,
                valid_acc,
                test_acc,
                train_time,
            ) = self.optimizer.train_statistics(report_incumbent)
            train_loss, valid_loss, test_loss = -1, -1, -1


            self.periodic_checkpointer.step(e)

            anytime_results = self.optimizer.test_statistics()


            self._log_to_json()

            self._log_and_reset_accuracies(e, summary_writer)

            if after_epoch is not None:
                after_epoch(e)

        logger.info(f"Saving architectural weight tensors: {self.config.save}/arch_weights.pt")
        if hasattr(self.config, "save_arch_weights") and self.config.save_arch_weights:
            torch.save(arch_weights, f'{self.config.save}/arch_weights.pt')
            if hasattr(self.config, "plot_arch_weights") and self.config.plot_arch_weights:
                plot_architectural_weights(self.config, self.optimizer)

        self.optimizer.after_training()

        if summary_writer is not None:
            summary_writer.close()

        logger.info("Training finished")



    def evaluate(
        self,
        retrain:bool=True,
        search_model:str="",
        resume_from:str="",
        best_arch:Graph=None,
        dataset_api:object=None,
        metric:Metric=None,
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool)      : Reset the weights from the architecure search
            search_model (str)  : Path to checkpoint file that was created during search. If not provided,
                                  then try to load 'model_final.pth' from search
            resume_from (str)   : Resume retraining from the given checkpoint file.
            best_arch           : Parsed model you want to directly evaluate and ignore the final model
                                  from the optimizer.
            dataset_api         : Dataset API to use for querying model performance.
            metric              : Metric to query the benchmark for.
        """
        logger.info("Start evaluation")
        if not best_arch:

            if not search_model:
                search_model = os.path.join(
                    self.config.save, "search", "model_final.pth"
                )
            self._setup_checkpointers(search_model)  # required to load the architecture

            best_arch = self.optimizer.get_final_architecture()
        logger.info(f"Final architecture hash: {best_arch.get_hash()}")

        if best_arch.QUERYABLE:
            if metric is None:
                metric = Metric.TEST_ACCURACY
            result = best_arch.query(
                metric=metric, dataset=self.config.dataset, dataset_api=dataset_api
            )
            logger.info("Queried results ({}): {}".format(metric, result))
            return result




    def _log_and_reset_accuracies(self, epoch, writer=None):
        logger.info(
            "Epoch {} done. Train accuracy: {:.5f}, Validation accuracy: {:.5f}".format(
                epoch,
                self.train_top1.avg,
                self.val_top1.avg,
            )
        )

        if writer is not None:
            writer.add_scalar('Train accuracy (top 1)', self.train_top1.avg, epoch)
            writer.add_scalar('Train accuracy (top 5)', self.train_top5.avg, epoch)
            writer.add_scalar('Train loss', self.train_loss.avg, epoch)
            writer.add_scalar('Validation accuracy (top 1)', self.val_top1.avg, epoch)
            writer.add_scalar('Validation accuracy (top 5)', self.val_top5.avg, epoch)
            writer.add_scalar('Validation loss', self.val_loss.avg, epoch)

        self.train_top1.reset()
        self.train_top5.reset()
        self.train_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_loss.reset()




