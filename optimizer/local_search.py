from .metabase import MetaOptimizer


class LocalSearch(MetaOptimizer):

    def __init__(self, config, dataset):
        """
        config:
            - num_init: number of initial models to sample before local search
            - query_full_nbhd: whether to query the full neighborhood of the current best architecture or stop at the first improvement.
        """
        super(LocalSearch, self).__init__(config, dataset)

        self.num_init = config.num_init
        self.query_full_nbhd = config.query_full_nbhd

        assert self.num_init <= self.total_queries

    def run(self):
        while self.queries < self.total_queries:
            # allow restart/retry if the number of queries is not met the termination condition
            stop_flag = self._local_search()
            if stop_flag:
                break

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

    def _local_search(self):
        """
        Return: True if the termination condition is met
        """
        best_arch = None
        best_metric = None
        init_model_cnt = 0

        while init_model_cnt < self.num_init:
            arch = self.search_space.sample_arch()
            if arch not in self.sampled_archs:
                arch_metric = self.query_metric(arch)
                init_model_cnt += 1

                if (
                    best_arch is None
                    or self.compare_metric(best_metric, arch_metric) > 0
                ):
                    best_arch = arch
                    best_metric = arch_metric

                if self.queries >= self.total_queries:
                    return True

        while True:
            improvement = False

            for neighbor in self.search_space.get_neighbors(best_arch, shuffle=True):
                if neighbor not in self.sampled_archs:
                    neighbor_metric = self.query_metric(neighbor)
                    if self.compare_metric(best_metric, neighbor_metric):
                        best_arch = neighbor
                        best_metric = neighbor_metric
                        improvement = True
                        if not self.query_full_nbhd:
                            break

                    if self.queries >= self.total_queries:
                        return True

            if not improvement:
                break
