from metabase import MetaOptimizer
import numpy as np
import random


class RegularizedEvolution(MetaOptimizer):

    def __init__(self, config, dataset):
        super(RegularizedEvolution).__init__(config, dataset)

        self.pop_size = config.pop_size
        self.tournament_size = config.tournament_size
        self.mutation_rate = config.mutation_rate
        self.regularized = config.regularized  # delete last or worst

        if self.pop_size > self.total_queries:
            self.pop_size = self.total_queries
            print(f"Warning: pop_size {self.pop_size} is larger than total_queries {self.total_queries}, \
                  set pop_size to {self.total_queries}")

    def run(self):
        pop = self.search_space.sample_archs(self.pop_size)
        pop_metric = [self.query_metric(arch) for arch in pop]
        iteration = 0
        pop_time = [iteration for _ in range(self.pop_size)]

        while self.queries < self.total_queries:
            iteration += 1
            parent_i = self.tournament_selection(pop_metric)
            parent = pop[parent_i]
            if random.random() < self.mutation_rate:
                child = self.mutate(parent)
                child_metric = self.query_metric(child)

                if self.regularized:
                    oldest_i = np.argmin(pop_time)
                    pop[oldest_i] = child
                    pop_metric[oldest_i] = child_metric
                    pop_time[oldest_i] = iteration
                else:
                    if self.objective_direction == "max":
                        worst_i = np.argmin(pop_metric)
                    elif self.objective_direction == "min":
                        worst_i = np.argmax(pop_metric)
                    else:
                        raise ValueError("Unknown objective direction.")
                    pop[worst_i] = child
                    pop_metric[worst_i] = child_metric
                    pop_time[worst_i] = iteration
                    

    def tournament_selection(self, pop_metric):
        sample_i = np.random.choice(len(pop_metric), self.tournament_size)
        sample_metric = [pop_metric[i] for i in sample_i]
        if self.objective_direction == "max":
            return sample_i[np.argmax(sample_metric)]
        elif self.objective_direction == "min":
            return sample_i[np.argmin(sample_metric)]
        else:
            raise ValueError("Unknown objective direction.")

    def mutate(self, arch):
        return self.search_space.get_random_neighbor(arch)
