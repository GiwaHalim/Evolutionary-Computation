from robust_cma_es import run_cma_es
from robust_genetic_algorithms import running_GA


algorithms = {
    "CMA-ES": {
        "name": "CMA-ES",
        "function": run_cma_es,
        "description": "Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a powerful optimization algorithm that adapts the covariance matrix of the search distribution to efficiently explore the search space"
    },
    "RCGA-P" : {
        "name": "RCGA-P",
        "function": running_GA,
        "description": "Robust Genetic Algorithm with Elitism and linear ranked  Selection."
}}