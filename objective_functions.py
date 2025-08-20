import numpy as np

class Ackley():
  @staticmethod
  def fitness_function(x):
    n = len(x)
    if n == 0:
        return 0.0
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)

    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return term1 + term2 + 20 + np.exp(1)

  @staticmethod
  def calculate_fitness(x):
    return abs(Ackley.fitness_function(x))

import numpy as np

class RotatedAckley:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedAckley.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedAckley.initialize_rotation_matrix(len(x))  # Ensure the rotation matrix is initialized
        if RotatedAckley.rotation_matrix is None:
            raise ValueError("Rotation matrix not initialized. Call initialize_rotation_matrix(dimension).")
        
        z = RotatedAckley.rotation_matrix @ x
        n = len(z)
        if n == 0:
            return 0.0
        sum1 = np.sum(z ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * z))

        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)

        return term1 + term2 + 20 + np.exp(1)

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedAckley.fitness_function(x))


class Sphere():
  @staticmethod
  def fitness_function(x):
    return sum(xi**2 for xi in x)

  @staticmethod
  def calculate_fitness(x):
    return abs(Sphere.fitness_function(x))

class Rosenbrock():
  @staticmethod
  def fitness_function(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

  @staticmethod
  def calculate_fitness(x):
    return abs(Rosenbrock.fitness_function(x))
  
class RotatedRosenbrock:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedRosenbrock.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedRosenbrock.initialize_rotation_matrix(len(x))
        z = RotatedRosenbrock.rotation_matrix @ x
        return np.sum(100 * (z[1:] - z[:-1]**2) ** 2 + (z[:-1] - 1) ** 2)

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedRosenbrock.fitness_function(x))

  
class HighConditionedElliptic:
    @staticmethod
    def fitness_function(x):
        d = len(x)
        return sum((10**6)**(i/(d-1)) * x[i]**2 for i in range(d))

    @staticmethod
    def calculate_fitness(x):
        return abs(HighConditionedElliptic.fitness_function(x))

class BentCigar:
    @staticmethod
    def fitness_function(x):
        return x[0]**2 + 10**6 * sum(xi**2 for xi in x[1:])

    @staticmethod
    def calculate_fitness(x):
        return abs(BentCigar.fitness_function(x))

class RotatedBentCigar:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedBentCigar.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedBentCigar.initialize_rotation_matrix(len(x))
        z = RotatedBentCigar.rotation_matrix @ x
        return z[0] ** 2 + 10 ** 6 * np.sum(z[1:] ** 2)

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedBentCigar.fitness_function(x))


class Discus:
    @staticmethod
    def fitness_function(x):
        return 10**6 * x[0]**2 + sum(xi**2 for xi in x[1:])

    @staticmethod
    def calculate_fitness(x):
        return abs(Discus.fitness_function(x))
    
class RotatedDiscus:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedDiscus.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedDiscus.initialize_rotation_matrix(len(x))
        z = RotatedDiscus.rotation_matrix @ x
        return 10 ** 6 * z[0] ** 2 + np.sum(z[1:] ** 2)

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedDiscus.fitness_function(x))


class DifferentPowers:
    @staticmethod
    def fitness_function(x):
        d = len(x)
        return sum(abs(x[i])**(2 + 4 * i / (d - 1)) for i in range(d))

    @staticmethod
    def calculate_fitness(x):
        return abs(DifferentPowers.fitness_function(x))

class SchaffersF7:
    @staticmethod
    def fitness_function(x):
        return (sum(np.sqrt(x[i]**2 + x[i+1]**2) + np.sqrt(x[i]**2 + x[i+1]**2) *
               (np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1)**2) for i in range(len(x)-1)) / (len(x)-1))**2

    @staticmethod
    def calculate_fitness(x):
        return abs(SchaffersF7.fitness_function(x))
class RotatedSchaffersF7:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedSchaffersF7.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedSchaffersF7.initialize_rotation_matrix(len(x))
        z = RotatedSchaffersF7.rotation_matrix @ x
        return np.mean([
            (np.sqrt(z[i] ** 2 + z[i + 1] ** 2) ** 0.5) *
            (np.sin(50 * (z[i] ** 2 + z[i + 1] ** 2) ** 0.1) ** 2 + 1)
            for i in range(len(z) - 1)
        ])

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedSchaffersF7.fitness_function(x))


class Weierstrass:
    @staticmethod
    def fitness_function(x):
        a, b, k_max = 0.5, 3, 20
        term1 = sum(sum(a**k * np.cos(2 * np.pi * b**k * (xi + 0.5)) for k in range(k_max + 1)) for xi in x)
        term2 = len(x) * sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
        return term1 - term2

    @staticmethod
    def calculate_fitness(x):
        return abs(Weierstrass.fitness_function(x))
    
class RotatedWeierstrass:
    rotation_matrix = None
    a, b, k_max = 0.5, 3, 20

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedWeierstrass.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedWeierstrass.initialize_rotation_matrix(len(x))
        z = RotatedWeierstrass.rotation_matrix @ x
        n = len(z)
        a, b, k_max = RotatedWeierstrass.a, RotatedWeierstrass.b, RotatedWeierstrass.k_max

        sum1 = np.sum([
            np.sum([a ** k * np.cos(2 * np.pi * b ** k * (z[i] + 0.5)) for k in range(k_max + 1)])
            for i in range(n)
        ])
        sum2 = n * np.sum([a ** k * np.cos(np.pi * b ** k) for k in range(k_max + 1)])
        return sum1 - sum2

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedWeierstrass.fitness_function(x))


class Griewank:
    @staticmethod
    def fitness_function(x):
        sum_part = sum(xi**2 for xi in x) / 4000
        prod_part = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
        return sum_part - prod_part + 1

    @staticmethod
    def calculate_fitness(x):
        return abs(Griewank.fitness_function(x))
    
class RotatedGriewank:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedGriewank.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedGriewank.initialize_rotation_matrix(len(x))
        z = RotatedGriewank.rotation_matrix @ x
        sum_sq = np.sum(z ** 2) / 4000
        prod_cos = np.prod([np.cos(z[i] / np.sqrt(i + 1)) for i in range(len(z))])
        return sum_sq - prod_cos + 1

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedGriewank.fitness_function(x))


class Rastrigin:
    @staticmethod
    def fitness_function(x):
        return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

    @staticmethod
    def calculate_fitness(x):
        return abs(Rastrigin.fitness_function(x))
    
class RotatedRastrigin:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedRastrigin.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedRastrigin.initialize_rotation_matrix(len(x))
        z = RotatedRastrigin.rotation_matrix @ x
        return 10 * len(z) + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z))

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedRastrigin.fitness_function(x))


class Schwefel:
    @staticmethod
    def fitness_function(x):
        return 418.9829 * len(x) - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)

    @staticmethod
    def calculate_fitness(x):
        return abs(Schwefel.fitness_function(x))

class Katsuura:
    @staticmethod
    def fitness_function(x):
        d = len(x)
        return np.prod([(1 + (i + 1) * sum(abs(2**j * xi - round(2**j * xi)) / 2**j for j in range(1, 33)))
                        ** (10 / d**1.2) for i, xi in enumerate(x)]) - 1

    @staticmethod
    def calculate_fitness(x):
        return abs(Katsuura.fitness_function(x))
    
class RotatedKatsuura:
    rotation_matrix = None

    @staticmethod
    def initialize_rotation_matrix(dimension, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(dimension, dimension)
        Q, _ = np.linalg.qr(A)
        RotatedKatsuura.rotation_matrix = Q

    @staticmethod
    def fitness_function(x):
        x = np.asarray(x)
        RotatedKatsuura.initialize_rotation_matrix(len(x))
        z = RotatedKatsuura.rotation_matrix @ x
        n = len(z)
        product = 1
        for i in range(n):
            sum_inner = 0
            for j in range(1, 33):
                sum_inner += abs(2 ** j * z[i] - round(2 ** j * z[i])) / 2 ** j
            product *= (1 + (i + 1) * sum_inner) ** (10 / n ** 1.2)
        return product - 1

    @staticmethod
    def calculate_fitness(x):
        return abs(RotatedKatsuura.fitness_function(x))


class LunacekBiRastrigin:
    @staticmethod
    def fitness_function(x):
        d = len(x)
        s, mu1, d_, mu2 = 1.0, 2.5, 1.0, -np.sqrt((2.5**2 - 1.0) / 2)
        term1 = sum((xi - mu1)**2 for xi in x)
        term2 = d * d_ + d_ * sum((xi - mu2)**2 for xi in x)
        term3 = 10 * sum(1 - np.cos(2 * np.pi * (xi - mu1)) for xi in x)
        return min(term1, term2) + term3

    @staticmethod
    def calculate_fitness(x):
        return abs(LunacekBiRastrigin.fitness_function(x))

class GriewankRosenbrock:
    @staticmethod
    def fitness_function(x):
        return Griewank.fitness_function([100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1)])

    @staticmethod
    def calculate_fitness(x):
        return abs(GriewankRosenbrock.fitness_function(x))

class SchaffersF6:
    @staticmethod
    def fitness_function(x):
        return sum(0.5 + (np.sin(np.sqrt(x[i]**2 + x[i+1]**2))**2 - 0.5) /
                   (1 + 0.001 * (x[i]**2 + x[i+1]**2))**2 for i in range(len(x)-1)) / (len(x)-1)

    @staticmethod
    def calculate_fitness(x):
        return abs(SchaffersF6.fitness_function(x))



objective_functions_configurations = {
    "Sphere": {
        "objective_function": Sphere,
        "bounds": [-5, 5],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 0.1
    },
    "Ackley": {
        "objective_function": Ackley,
        "bounds": [-32, 32],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 0.5
    },
    "RotatedAckley": {
        "objective_function": RotatedAckley,
        "bounds": [-32, 32],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 0.5
    },
    "Rosenbrock": {
        "objective_function": Rosenbrock,
        "bounds": [-5, 5],
        "minimum": 1,
        "value_at_minimum": 0,
        "step_size": 0.1
    },
    "RotatedRosenbrock": {
        "objective_function": RotatedRosenbrock,
        "bounds": [-5, 5],
        "minimum": 1,
        "value_at_minimum": 0,
        "step_size": 0.1
    },
    "HighConditionedElliptic": {
        "objective_function": HighConditionedElliptic,
        "bounds": [-5, 5],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 2.0
    },
    "BentCigar": {
        "objective_function": BentCigar,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 50.0
    },
    "RotatedBentCigar": {
        "objective_function": RotatedBentCigar,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1.0
    },
    "Discus": {
        "objective_function": Discus,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 50.0
    },
    "RotatedDiscus": {
        "objective_function": RotatedDiscus,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 50.0
    },
    "DifferentPowers": {
        "objective_function": DifferentPowers,
        "bounds": [-1, 1],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 0.01
    },
    "SchaffersF7": {
        "objective_function": SchaffersF7,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1.0
    },
    "RotatedSchaffersF7": {
        "objective_function": RotatedSchaffersF7,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1.0
    },
    "Weierstrass": {
        "objective_function": Weierstrass,
        "bounds": [-0.5, 0.5],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 0.005
    },
    "RotatedWeierstrass": {
        "objective_function": RotatedWeierstrass,
        "bounds": [-0.5, 0.5],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 0.005
    },
    "Griewank": {
        "objective_function": Griewank,
        "bounds": [-600, 600],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 100.0
    },
    "RotatedGriewank": {
        "objective_function": RotatedGriewank,
        "bounds": [-600, 600],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 100.0
    },
    "Rastrigin": {
        "objective_function": Rastrigin,
        "bounds": [-5.12, 5.12],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1.0
    },
    "RotatedRastrigin": {
        "objective_function": RotatedRastrigin,
        "bounds": [-5.12, 5.12],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 0.1
    },
    "Schwefel": {
        "objective_function": Schwefel,
        "bounds": [-500, 500],
        "minimum": 420.9687,
        "value_at_minimum": 0,
        "step_size": 100
    },
    "Katsuura": {
        "objective_function": Katsuura,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1.0
    },
    "RotatedKatsuura": {
        "objective_function": RotatedKatsuura,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1.0
    },
    "LunacekBiRastrigin": {
        "objective_function": LunacekBiRastrigin,
        "bounds": [-5, 5],
        "minimum": 2.5,
        "value_at_minimum": 0,
        "step_size": 1
    },
    "GriewankRosenbrock": {
        "objective_function": GriewankRosenbrock,
        "bounds": [-5, 5],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1
    },
    "SchaffersF6": {
        "objective_function": SchaffersF6,
        "bounds": [-100, 100],
        "minimum": 0,
        "value_at_minimum": 0,
        "step_size": 1.0
    }
}