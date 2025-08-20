import numpy as np

class RCGA_P:
  @staticmethod
  def linear_ranked_selection(population, num_parents_to_select, fitness):
    N = len(population)  #length of population
    # Sort indices based on fitness in ascending order (highest fitness first)
    ranked_indices = np.argsort(fitness)[::-1]

    # Linear rank selection formula
    probs = 1.1 - ((2 * (1.1 - 1) * (ranked_indices)) / (N - 1))
    probs /= probs.sum()

    # Select parents based on the calculated probabilities with replacement, we are using numpy random to select based on propability, we are choosing two random parents.

    # This returns the indeces of the parents
    selected_indices = np.random.choice(N, num_parents_to_select, p=probs, replace=True)
    # We select the parents
    selected_parents = population[selected_indices]

    #We return the selected parents to have the crossed and mutated based on probability
    return selected_parents

  @staticmethod
  def arithmetic_crossover(parents, crossover_rate, bounds):
    offspring = []
    parent1 = parents[0]
    parent2 = parents[1]

    child1 = []
    child2 = []

    # Perform crossover with a given probability
    if np.random.rand() < crossover_rate:
        for i in range(len(parent1)):
            # Alpha for arithmetic crossover, allowing for extrapolation
            alpha = np.random.uniform(-0.5, 1.5)
            # Create child1's gene
            child1Cross = alpha * parent1[i] + (1 - alpha) * parent2[i]
            child1.append(np.clip(child1Cross, bounds[0], bounds[1]))
            child2Cross = alpha * parent2[i] + (1 - alpha) * parent1[i]
            child2.append(np.clip(child2Cross, bounds[0], bounds[1]))
    else:
        # If no crossover, children are exact copies of parents
        child1 = list(parent1) # Create copies to avoid modifying original parents
        child2 = list(parent2)

    offspring.append(child1)
    offspring.append(child2)

    return offspring
  @staticmethod
  def mutation(individuals, p_mu, bounds, mutation_strength=0.1):
    mutated_individuals = []
    for individual in individuals: # Iterate through each individual in the list
        mutated_individual = list(individual) # Create a mutable copy of the individual
        for i in range(len(mutated_individual)): # Iterate through each dimension (gene) of the individual
            if np.random.rand() < p_mu:
                # Apply mutation: add a random value between -mutation_strength and +mutation_strength
                mutated_individual[i] += np.random.uniform(-mutation_strength, mutation_strength)
                # Clip the mutated value to ensure it stays within the search space bounds
                mutated_individual[i] = np.clip(mutated_individual[i], bounds[0], bounds[1])
        mutated_individuals.append(mutated_individual)
    return mutated_individuals

  @staticmethod
  def vector_projection(u, v):
    """Project vector u onto vector v."""
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)

    norm_sq = np.dot(v, v)
    if norm_sq == 0:
        return np.zeros_like(v)  # Avoid division by zero
    scalar = np.dot(u, v) / norm_sq
    return scalar * v

  @staticmethod
  def projection_based_generation(M_t, fitness_func):
    M_t = np.array(M_t, dtype=float)  # Ensure it's a NumPy array
    m = len(M_t)
    new_population = []

    for _ in range(m):
        i, j = np.random.randint(0, m, size=2)
        x = M_t[i]
        y = M_t[j]

        fx = fitness_func(x)
        fy = fitness_func(y)

        if fx < fy:
            s = RCGA_P.vector_projection(x, y)  # project x onto y
        else:
            s = RCGA_P.vector_projection(y, x)  # project y onto x

        new_population.append(s)

    return np.array(new_population)
