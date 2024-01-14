from metabase import MetaOptimizer




class LocalSearch(MetaOptimizer):

    def __init__(self, config, dataset):
        super(LocalSearch, self).__init__(config, dataset)

        self.num_init = config.search.num_init
        self.query_full_nbhd = config.search.query_full_nbhd

    def run(self):
        while self.queries < self.total_queries:
            stop_flag = self._local_search()
            if stop_flag:
                break
            
    def _local_search(self):
        """
            Return: True if the termination condition is met
        """
        best_arch = self.search_space.sample_arch()
        best_metric = self.dataset.get_objective(
                        arch, self.objective_metric)
        init_model_cnt = 1
        while init_model_cnt < self.num_init:
            arch = self.search_space.sample_arch()
            if arch not in self.sampled_archs:
                arch_metric = self.dataset.get_objective(
                        arch, self.objective_metric)
                init_model_cnt += 1
                if self.compare_metric(best_metric, arch_metric):
                    best_arch = arch
                    best_metric = arch_metric
                if self.queries >= self.total_queries:
                    return True
                    
        while True:
            improvement = False

            for neighbor in self.search_space.get_neighbors(best_arch):
                if neighbor not in self.sampled_archs:
                    neighbor_metric = self.dataset.get_objective(
                        neighbor, self.objective_metric)
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


