import numpy as np
import torch

from search_space import OneHotCodec
from ..metabase import MetaOptimizer
from .predictor import Ensemble
from .acq_fn import acq_fn


class Bananas(MetaOptimizer):
    def __init__(self, config, dataset):
        """
        Initialize a random search optimizer.

        Args:
            config: Config file
        """
        super(Bananas, self).__init__(config, dataset)

        self.num_init = config.num_init
        self.k = config.k

        self.acq_opt_type = config.acq_opt_type
        self.explore_type = config.explore_type

        self.num_candidates = config.num_candidates
        self.num_arches_to_mutate = config.num_arches_to_mutate
        self.mutations = config.mutations  # max number of muatations for a parent
        self.patience_factor = config.patience_factor

        if not torch.cuda.is_available():
            config.device = "cpu"

        self.device = torch.device(config.device)

        self.codec = OneHotCodec(self.search_space)

        self.predictor_cfg = config.predictor

        assert self.acq_opt_type in ["mutation", "mutation_random",
                                     "random"], f"{self.acq_opt_type} is not yet implemented as an acquisition type"

        assert "acc" in config.objective_metric.lower(
        ), "Only Acc metrics are supported in BANANAS, eg: VAL_PGD_ACC"

    def run(self):
        """
        Run the optimizer.
        """

        for arch in self.search_space.sample_archs(self.num_init):
            # results are saved in self.sampled_archs
            self.query_metric(arch)

        while self.queries < self.total_queries:
            xtrain = []
            ytrain = []
            for arch, metric in self.sampled_archs.items():
                xtrain.append(self.codec.encode(arch))
                # Note: here we need val_error, which is 1-val_acc
                ytrain.append(100-metric)

            xtrain = np.stack(xtrain, axis=0)
            ytrain = np.array(ytrain)

            predictor = Ensemble(self.predictor_cfg, self.device)
            predictor.fit(xtrain, ytrain)

            candidate_archs = self.get_candidates()
            xcandidates = [self.codec.encode(arch)
                           for arch in candidate_archs]
            xcandidates = np.stack(xcandidates, dtype=np.float32)

            candidate_predictions = predictor.query(xcandidates)
            acq_val = acq_fn(
                candidate_predictions, ytrain=ytrain, explore_type=self.explore_type
            )
            candidate_indices = np.argsort(acq_val)

            new_cnt = 0
            for i in candidate_indices[:self.k]:
                arch = self.codec.decode(xcandidates[i])
                if arch not in self.sampled_archs:
                    new_cnt += 1

                self.query_metric(arch)

            print(f"Queries: {self.queries}, new: {new_cnt}")

        if self.objective_direction == "max":
            self.best_arch_metric = max(self.sampled_archs.values())
        elif self.objective_direction == "min":
            self.best_arch_metric = min(self.sampled_archs.values())
        else:
            raise ValueError("Unknown objective direction.")

        self.best_archs = [
            arch
            for arch, metric in self.sampled_archs.items()
            if metric == self.best_arch_metric
        ]

    def get_candidates(self):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        Note: when acq_opt_type = random or mutation_random, generate (2 * self.num_candidates) candidates, following the impl of original BANANAS
        """

        candidates = []

        if self.acq_opt_type in ["mutation", "mutation_random"]:
            # mutate architectures with the lowest loss (i.e., highest acc)
            best_arches = self.get_topk_archs(
                k=self.num_arches_to_mutate*self.patience_factor)

            # stop when candidates is size num

            for arch in best_arches:
                if len(candidates) >= self.num_candidates:
                    break
                for i in range(
                    int(
                        self.num_candidates
                        / self.num_arches_to_mutate
                        / self.mutations
                    )
                ):
                    for j in range(self.mutations):
                        new_arch = self.mutate(arch)

                        if new_arch not in self.sampled_archs:
                            candidates.append(new_arch)

        if self.acq_opt_type in ["random", "mutation_random"]:
            # add randomly sampled architectures to the set of candidates
            for i in range(self.num_candidates * self.patience_factor):
                if len(candidates) >= 2 * self.num_candidates:
                    break
                new_arch = self.search_space.sample_arch()
                if new_arch not in self.sampled_archs:
                    candidates.append(new_arch)

        return candidates

    def mutate(self, arch):
        return self.search_space.get_random_neighbor(arch)
