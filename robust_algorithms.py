from robust_cma_es import run_cma_es
from robust_genetic_algorithms_with_projection import running_RCGA_P


algorithms = {
    "CMA-ES": {
        "name": "CMA-ES",
        "function": run_cma_es,
        "description": "Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a powerful optimization algorithm that adapts the covariance matrix of the search distribution to efficiently explore the search space"
    },
    "RCGA-P" : {
        "name": "RCGA-P",
        "function": running_RCGA_P,
        "description": "Robust Genetic Algorithm with projection with Elitism and linear ranked  Selection."
}}