import numpy as np


def generate(num):
    return np.random.randint(8, size=(num, 8))


def fitness(chromosome):
    attacking = 0

    for i in range(8):
        for j in range(i+1, 8):
            if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == abs(i - j):
                attacking += 1

    return 28 - attacking


def fitness_percent(chromosomes):
    fit = np.apply_along_axis(fitness, axis=1, arr=chromosomes)
    return fit / sum(fit)


def selection(chromosomes):
    selected = []
    percent_fit = fitness_percent(chromosomes)
    cdfs = np.cumsum(percent_fit)

    for _ in range(len(chromosomes)):
        parents = []

        while len(parents) < 2:
            rand = np.random.random()
            x = np.argwhere(cdfs > rand)[0][0]

            if x in parents:
                continue

            parents.append(x)

        selected.append(parents)

    return selected


def crossover(parent1, parent2):
    children = []
    rand = np.random.randint(0, 8, dtype=int)

    child1 = np.concatenate((parent1[:rand], parent2[rand:]))
    child2 = np.concatenate((parent2[:rand], parent1[rand:]))

    children.append(mutate(child1))
    children.append(mutate(child2))

    choice = 1 if np.random.rand() > 0.5 else 0
    return children[choice]


def evolve(chromosomes):
    chromosomes_new = []
    mating_pairs = selection(chromosomes)

    for pair in mating_pairs:
        chromosomes_new.append(
            crossover(chromosomes[pair[0]], chromosomes[pair[1]]
                      ))

    return np.array(chromosomes_new)


def mutate(chromosome):
    gene = np.random.randint(0, 8, dtype=int)
    rand = np.random.randint(0, 8, dtype=int)

    chromosome[gene] = rand

    return chromosome


def simulate(population, iters, max_iters=1000, custom_chromosomes=None):
    if custom_chromosomes is not None:
        chromosomes = custom_chromosomes
    else:
        chromosomes = generate(population)

    print(f'Initial Generation:\n{chromosomes}\n')

    gen = 0
    index = None
    while gen < max_iters:
        fit = np.apply_along_axis(fitness, axis=1, arr=chromosomes)

        if gen % 100 == 0:
            print(f'{gen}: Max fitness = {max(fit)}')

        try:
            index = np.where(fit == 28)[0][0]
            break
        except:
            index = None

        chromosomes = evolve(chromosomes)
        gen += 1

    if index is not None:
        print(f'solution: {chromosomes[index]}')
    else:
        print('solution not found!')


chromosomes = np.array([[0, 4, 7, 5, 1, 3, 6, 2],
                        [1, 3, 0, 4, 6, 2, 5, 7],
                        [2, 0, 4, 7, 1, 3, 5, 6]
                        ])
simulate(20, 100, 10000)
