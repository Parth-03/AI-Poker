
import random
import copy
from pypokerengine.api.game import setup_config, start_poker

class Individual:
    def __init__(self, weights):
        self.weights = weights
        self.fitness = None

    def mutate(self, mutation_rate, mutation_strength=0.05):
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += random.uniform(-mutation_strength, mutation_strength)

    @staticmethod
    def crossover(parent1, parent2, method='average'):
        if method == 'average':
            child_weights = [(w1 + w2) / 2 for w1, w2 in zip(parent1.weights, parent2.weights)]
        else:
            child_weights = [random.choice([w1, w2]) for w1, w2 in zip(parent1.weights, parent2.weights)]
        return Individual(child_weights)

class Population:
    def __init__(self, size, num_features):
        self.individuals = [Individual([random.uniform(-1, 1) for _ in range(num_features)]) for _ in range(size)]
        self.generation = 0

    def evaluate(self, evaluator):
        for individual in self.individuals:
            individual.fitness = evaluator.evaluate_fitness(individual)

    def select_parents(self, tournament_size=5):
        tournament = random.sample(self.individuals, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0], tournament[1]

    def evolve(self, elite_size, crossover_rate, mutation_rate):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        new_population = self.individuals[:elite_size]
        while len(new_population) < len(self.individuals):
            parent1, parent2 = self.select_parents()
            if random.random() < crossover_rate:
                child = Individual.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            child.mutate(mutation_rate)
            new_population.append(child)
        self.individuals = new_population
        self.generation += 1

class Evaluator:
    def __init__(self, baseline_agents, weighting_factors):
        self.baseline_agents = baseline_agents
        self.weighting_factors = weighting_factors

    def evaluate_fitness(self, individual):
        from agent import CustomAgent
        agent = CustomAgent(individual.weights)
        total_fitness = 0
        for baseline_name, opponent_agent in self.baseline_agents.items():
            result = simulate_match(agent, opponent_agent)
            weighted_result = result * self.weighting_factors[baseline_name]
            total_fitness += weighted_result
        return total_fitness

def simulate_match(agent, opponent_agent, num_rounds=50):
    config = setup_config(max_round=num_rounds, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="agent", algorithm=agent)
    config.register_player(name="opponent", algorithm=opponent_agent)
    game_result = start_poker(config, verbose=0)
    stack_agent = [s['stack'] for s in game_result['players'] if s['name'] == "agent"][0]
    return (stack_agent - 1000) / 1000
