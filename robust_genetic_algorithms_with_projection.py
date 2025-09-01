import numpy as np
import random

# from objective_functions import objective_functions_configurations
from genetic_algorithm_with_projection import RCGA_P

from objective_functions import objective_functions_configurations



# Define GA parameters
# OBJECTIVE_FUNCTION = "Sphere"  # Choose from "Ackley", "Sphere", "Rosenbrock"
# Constants for the Genetic Algorithm with projection
# These can be adjusted based on the problem and desired performance
# Population size, number of dimensions, bounds for the objective function, etc.
POPULATION_SIZE = 500
GENERATIONS = 1000
CROSSOVER_RATE = 0.6 # Probability of crossover occurring
MUTATION_RATE = 0.001 # Probability of a single gene being mutated
ELITISM_COUNT = 1 # Number of best individuals to carry over

def create_individual(num_dimensions, bounds):
    return [random.uniform(bounds[0], bounds[1]) for _ in range(num_dimensions)]


def running_RCGA_P(
    calculate_fitness,
    OBJECTIVE_FUNCTION,
    pop_size,
    num_dimensions,
    bounds,
    generations,
    crossover_rate,
    mutation_rate,
    elitism_count=1):

  population = [create_individual(num_dimensions, bounds) for _ in range(pop_size)]
  best_overall_individual = None
  best_overall_fitness = float('inf') # Initialize with a very low fitness value
  best_fitness_history = []

  function_calls = 0
  print(f"\nStarting Genetic Algorithm with projection for {OBJECTIVE_FUNCTION} Function")

  for generation in range(generations):
    # Evaluate fitness for all individuals in the current population
    current_population_fitness = [(ind, calculate_fitness(ind)) for ind in population]
    #Tracking function evaluations
    function_calls += len(current_population_fitness)
    # Sort the population by fitness in descending order (best fitness first)
    current_population_fitness.sort(key=lambda item: item[1],)



    # Update the best overall individual found so far
    current_best_individual_gen = current_population_fitness[0][0]
    current_best_fitness_gen = current_population_fitness[0][1]
    best_fitness_history.append(current_best_fitness_gen)

    if current_best_fitness_gen < best_overall_fitness:
      best_overall_fitness = current_best_fitness_gen
      # Store a copy of the best individual's genes
      best_overall_individual = list(current_best_individual_gen)

     # Initialize the new generation list
    new_population = []

    # Apply Elitism: Carry over the best 'elitism_count' individuals directly.
    # This implicitly satisfies "If m = N, then elitism is applied" because we are generating a full new population of size N (pop_size),
    if elitism_count > 0:
      for i in range(elitism_count):
        new_population.append(list(current_population_fitness[i][0])) # Add a copy of the elite


    # Select m = N solutions from Pt as parents using linear-ranked selection to form a mating pool P_hat.
    fitness = [item[1] for item in current_population_fitness]
    population = np.array(population)
    mating_pool = RCGA_P.linear_ranked_selection(population, pop_size, fitness)
    # Convert the mating pool to a list to use the .pop() method
    mating_pool = mating_pool.tolist()
    # Shuffle the mating pool to randomize the order for sequential selection
    random.shuffle(mating_pool)

    # Generate the rest of the new population through selection, crossover, and mutation after appkying elitism
    num_offspring_to_generate = pop_size - elitism_count
    num_crossovers_needed = num_offspring_to_generate // 2 # Integer division

    for _ in range(num_crossovers_needed):
      # 4. Select pairs of parents sequentially from P_hat and use arithmetic crossover
      #    with probability ptc to create offspring solutions. Save the offspring in Ct.
      # We take two parents sequentially from the shuffled mating_pool.
      # If the mating_pool runs out, we can refill it or handle the edge case.
      if len(mating_pool) < 2:
        # If mating pool runs out, re-select a new pool or shuffle existing.
        # For robustness, let's re-select a new mating pool if it's too small.
        mating_pool = RCGA_P.select_parents(population, pop_size)
        mating_pool = mating_pool.tolist() # Convert to list
        random.shuffle(mating_pool)

      parent1 = mating_pool.pop(0) # Get first parent
      parent2 = mating_pool.pop(0) # Get second parent
      selected_parents = np.array([parent1, parent2])

      crossed_parents = RCGA_P.arithmetic_crossover(selected_parents, crossover_rate, bounds)

      # 5. Using random mutation, perform mutation on each component of yi,t ∈ Ct
      #  with a low probability, pμ, to create Mt.
      mutated_parents = RCGA_P.mutation(crossed_parents, mutation_rate, bounds)

      projected_mutated_parents = RCGA_P.projection_based_generation(mutated_parents, calculate_fitness)

      new_population.extend(projected_mutated_parents)



      # Add the mutated offspring to the new population
      # new_population.extend(mutated_parents)

    # 6. Update Pt by replacing m solutions in Pt with the solutions in Mt to create Pt+1.
    # Since we've built a new_population of size 'pop_size' (N), this is a full generational replacement.
    population = np.array(new_population[:pop_size]) # Ensure exactly pop_size individuals

    min_value = calculate_fitness(best_overall_individual)

    if min_value < 10**-6:
      print(f"Early stopping at generation {generation + 1} with fitness {best_overall_fitness:.6f}")
      break

    # Print progress for the current generation
    # print(f"Generation {generation + 1}/{generations}:")
    # print(f"  Best individual in this generation: {np.round(current_best_individual_gen, 4)}")
    # print(f"  Fitness (negative {OBJECTIVE_FUNCTION} value): {current_best_fitness_gen:.6f}")
    # print("-" * 30)

  # Return the best overall solution found and its Ackley function value
  print(f"Final Best Result:")
  print(f"Solution: {np.round(best_overall_individual, 6)}")
  print(f"{OBJECTIVE_FUNCTION}'s Function Value: {min_value:.6f}")
  print(f"Expected Minimum is 0 at: {[objective_functions_configurations[OBJECTIVE_FUNCTION]["minimum"]] * num_dimensions}")
  return best_overall_individual, min_value, best_fitness_history, function_calls


