




from ..metabase import MetaOptimizer
import numpy as np
import torch

from search_space import OneHotCodec
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
        self.max_mutations = config.max_mutations

        if not torch.cuda.is_available():
            config.device = "cpu"

        self.device = torch.device(config.device)

        self.codec = OneHotCodec(self.search_space)

        self.predictor_cfg = config.predictor

        if self.acq_opt_type not in ["mutation", "mutation_random", "random"]:
            raise NotImplementedError(
                f"{self.acq_opt_type} is not yet implemented as an acquisition type"
            )

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
                ytrain.append(metric)

            xtrain = np.stack(xtrain, axis=0)
            ytrain = np.array(ytrain)

            predictor = Ensemble(self.predictor_cfg, self.device)
            predictor.fit(xtrain, ytrain)

            xcandidates = [self.codec.encode(arch) for arch in self.get_candidates()]
            xcandidates = np.stack(xcandidates, dtype=np.float32)

            candidate_predictions = predictor.query(xcandidates)
            candidate_indices = acq_fn(
                candidate_predictions, ytrain=ytrain, explore_type=self.explore_type
            )

            new_cnt = 0
            for i in candidate_indices[: self.k]:
                arch = self.codec.decode(xcandidates[i])
                self.query_metric(arch)

                if arch not in self.sampled_archs:
                    new_cnt += 1
            
            print(f"Queries: {self.queries}, new: {new_cnt}")

    def get_candidates(self):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """

        candidates = []

        if self.acq_opt_type in ["mutation", "mutation_random"]:
            # mutate architectures with the lowest loss
            best_arches = self.get_topk_archs(k=self.num_arches_to_mutate)

            # stop when candidates is size num

            for arch in best_arches:
                if len(candidates) >= self.num_candidates:
                    break
                for i in range(
                    int(
                        self.num_candidates
                        / self.num_arches_to_mutate
                        / self.max_mutations
                    )
                ):
                    for j in range(self.max_mutations):
                        arch = self.mutate(arch)

                        if arch not in self.sampled_archs:
                            candidates.append(arch)

        if self.acq_opt_type in ["random", "mutation_random"]:
            # add randomly sampled architectures to the set of candidates
            for i in range():
                arch = self.search_space.sample_arch()
                if arch not in self.sampled_archs:
                    candidates.append(arch)

        return candidates

    def mutate(self, arch):
        return self.search_space.get_random_neighbor(arch)
