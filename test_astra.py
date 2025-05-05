import pytest
from Astra import GeneticAlgorithm, q_learning_difficulty_adjustment

def test_genetic_algorithm_evaluation():
    ga = GeneticAlgorithm()
    specimen = ga.population[0]
    ga.evaluate_fitness(specimen, 10, True)
    assert specimen["fitness"] == 20

def test_q_learning_adjustment_logic():
    _, state, action = q_learning_difficulty_adjustment(5, 1)
    assert action in ["increase", "decrease", "maintain"]
    assert isinstance(state, tuple)
    assert len(state) == 2
