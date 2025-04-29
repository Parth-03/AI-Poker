
from evolution import Population, Evaluator
from agent import CustomAgent
import random

def train_evolutionary_agent():
    population_size = 50
    num_features = 8
    num_generations = 200
    elite_size = 2
    crossover_rate = 0.7
    mutation_rate = 0.05

    baseline_agents = {
        'naive_mc': CustomAgent([random.uniform(-1, 1) for _ in range(num_features)]),
        'random': CustomAgent([random.uniform(-1, 1) for _ in range(num_features)]),
        'raise': CustomAgent([random.uniform(-1, 1) for _ in range(num_features)])
    }

    weighting_factors = {
        'naive_mc': 0.7,
        'random': 0.15,
        'raise': 0.15
    }

    evaluator = Evaluator(baseline_agents, weighting_factors)
    population = Population(population_size, num_features)

    for generation in range(num_generations):
        print(f"Generation {generation}")
        population.evaluate(evaluator)
        best_fitness = max(ind.fitness for ind in population.individuals)
        print(f"Best Fitness: {best_fitness}")
        population.evolve(elite_size, crossover_rate, mutation_rate)

    population.individuals.sort(key=lambda x: x.fitness, reverse=True)
    best_individual = population.individuals[0]
    return CustomAgent(best_individual.weights)

if __name__ == "__main__":
    trained_agent = train_evolutionary_agent()
    print("Training complete.")
