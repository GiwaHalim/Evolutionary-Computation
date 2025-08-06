import numpy as np
from objective_functions import objective_functions_configurations

from cma_es import CMA_ES

def run_cma_es(objective_func, OBJECTIVE_FUNCTION, bounds,  N=10, gen=1000,  sigma=0.5 ):
    print(f"\nStarting CMA-ES for {OBJECTIVE_FUNCTION} Function")

    cma_es = CMA_ES(objective_func, OBJECTIVE_FUNCTION, bounds, N, gen, sigma)
    best_solution, min_value, fitness_history, function_evaluations = cma_es.run()

    print(f"Final Best Result:")
    print(f"Solution: {np.round(best_solution, 6)}")
    print(f"{OBJECTIVE_FUNCTION}'s Function Value: {min_value:.6f}")
    print(f"Expected Minimum is 0 at: {[objective_functions_configurations[OBJECTIVE_FUNCTION]["minimum"]] * N}")

    return best_solution, min_value, fitness_history, function_evaluations
