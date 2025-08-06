import numpy as np

class CMA_ES:

    def __init__(self, objective_func, OBJECTIVE_FUNCTION, bounds,  N=10,  gen=1000, sigma=0.3, ):
        self.OBJECTIVE_FUNCTION = OBJECTIVE_FUNCTION
        self.objective_func = objective_func
        self.N = N
        self.sigma = sigma
        self.gen = gen
        self.bounds = bounds

        # Selection parameters
        self.lambda_ = 4 + int(3 * np.log(N))  # Population size
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = (np.sum(self.weights) ** 2) / np.sum(self.weights ** 2)

        # Adaptation parameters
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs

        # Initialization
        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.B = np.eye(N)
        self.D = np.ones(N)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D ** -1) @ self.B.T
        self.chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        self.counteval = 0

    def run(self):
        xmean = np.random.rand(self.N)  # Initial mean

        best_overall_individual = None
        best_overall_fitness = float('inf') # Initialize with a very high fitness value
        best_fitness_history = []

        for generation in range(self.gen):
            # Generate and evaluate offspring
            arz = np.random.randn(self.N, self.lambda_)
            ary = self.B @ (self.D.reshape(-1, 1) * arz)
            arx = xmean.reshape(-1, 1) + self.sigma * ary
            arfitness = np.apply_along_axis(self.objective_func, 0, arx)

            # Apply bounds
            lower_bound, upper_bound = self.bounds
            arx = np.clip(arx, lower_bound, upper_bound)

            self.counteval += self.lambda_

            # Sort by fitness
            idx = np.argsort(arfitness)
            xold = xmean.copy()
            xmean = arx[:, idx[:self.mu]] @ self.weights
            zmean = arz[:, idx[:self.mu]] @ self.weights

            # updating the best individual
            best_index = idx[0]
            current_best_fitness_gen = arfitness[best_index]
            current_best_individual_gen = arx[:, best_index]
            best_fitness_history.append(current_best_fitness_gen)

            if current_best_fitness_gen < best_overall_fitness:
                best_overall_fitness =  current_best_fitness_gen
                # Store a copy of the best individual's genes
                best_overall_individual = list(current_best_individual_gen)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ zmean)
            hsig = int(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lambda_)) / self.chiN < 1.4 + 2 / (self.N + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.B @ zmean)

            # Update covariance matrix
            artmp = (1 / self.sigma) * (arx[:, idx[:self.mu]] - xold.reshape(-1, 1))
            self.C = (1 - self.c1 - self.cmu) * self.C + \
                     self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                     self.cmu * artmp @ np.diag(self.weights) @ artmp.T

            # Update step-size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            # Decomposition of C
            if self.counteval - self.lambda_ * np.floor(self.counteval / self.lambda_) < 1:
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(np.maximum(self.D, 1e-20))   

            if -abs(best_overall_fitness - 0)  > -10**-6:
                print(f"Early stopping at generation {generation + 1} with fitness {best_overall_fitness:.6f}")
                break

            # best_fitness = arfitness[idx[0]]
            # print(f"Generation {generation + 1}/{self.gen}:")
            # print(f"  Best individual in this generation: {np.round(current_best_individual_gen, 4)}")
            # print(f"  Fitness (negative {self.OBJECTIVE_FUNCTION} value): {arfitness[idx[0]]:.6f}")
            # print("-" * 30)
        

        return best_overall_individual, best_overall_fitness, best_fitness_history, self.counteval

    

