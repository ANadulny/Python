import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import collections
import pyclipper
from hospitals.common import relative_path, get_csv_lines
from hospitals.initialData import CitiesDataCleaner, filter_input_data
from hospitals.gmaps import Map
from itertools import chain
import sys
from functools import lru_cache
import math

from typing import Iterable, Union

City = collections.namedtuple('City', ['id_number', 'towns', 'population'])

in_filename = 'spis_miast_2018.csv'


@lru_cache(maxsize=None)
def _area(shape, is_coord=False):
    return abs(sum(x1 * y2 - y1 * x2
                   for (x1, y1), (x2, y2)
                   in zip(shape, chain(shape[1:], shape[:1]))) / 2) * (1 if not is_coord else 111111 ** 2)


class Fenotype(tuple):

    scale = 1_000_000

    def __new__(cls, fenotype: Iterable, *args, **kwargs):
        return super(Fenotype, cls).__new__(cls, fenotype)

    def __init__(self, _=None, radius: Union[int, float] = 93.75*1000, *args, **kwargs):
        self.radius = radius
        self.__score = None

    @property
    def shapes(self):
        # None is a workaround for it being a instance method while it could have been static method
        return [Map.get_cycle(None, loc.lat, loc.lng, self.radius) for loc in self]

    def _shape(self, country: str = 'Poland'):
        border = Map.country_border(country)
        pc = pyclipper.Pyclipper()
        pc.AddPath(path=self.__to_pyclipper(border), poly_type=pyclipper.PT_SUBJECT, closed=True)
        for shape in self.shapes:
            pc.AddPath(path=self.__to_pyclipper(shape), poly_type=pyclipper.PT_CLIP, closed=True)

        solution = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        return tuple(self.__from_pyclipper(sol) for sol in solution)

    @property
    def area(self):
        return sum(_area(shape, True) for shape in self._shape())


    @classmethod
    def __to_pyclipper(cls, shape):
        return [(int(lat*cls.scale), int(lng*cls.scale)) for lat, lng in shape]

    @classmethod
    def __from_pyclipper(cls, shape):
        return tuple((lat / cls.scale, lng / cls.scale) for lat, lng in shape)

    @property
    def score(self):
        if self.__score is None:
            self.__score = 1 / (len(self) + 1000) + 1000000 / (self.area + 1)
        return self.__score


class Genotype(tuple):
    __city_data = []

    def __new__(cls, genotype: Iterable = None):
        """
        :param genotype: Iterable of booleans representing whether city is used or not
        """
        if genotype is None:
            genotype = tuple(int(random.getrandbits(1)) for _ in cls.get_city_data())

        assert len(genotype) == len(cls.get_city_data())

        return super(Genotype, cls).__new__(cls, genotype)
        # self.mutable = [int(random.getrandbits(1)) for _ in self.city_data]

    @classmethod
    def set_city_data(cls, city_data):
        cls.__city_data = [City(**city) for city in city_data]

    @classmethod
    def get_city_data(cls):
        if not cls.__city_data:
            data = get_csv_lines(relative_path(__file__, in_filename))
            cleaned_data = CitiesDataCleaner.clean(data)
            cls.__city_data = filter_input_data(cleaned_data)
        return cls.__city_data

    @property
    def fenotype(self):
        return Fenotype(Map.locations()[city['towns']] for city, mutable in zip(self.get_city_data(), self) if mutable)

    @property
    def score(self):
        return self.fenotype.score


def progress_bar(iterable, msg="Used {} elements!"):
    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    iter_len = len(iterable)
    for index, item in enumerate(iterable, 1):
        progress(index, iter_len, msg.format(index))
        if index == iter_len:
            print()
        yield item


def create_this():
    cityList = Genotype()
    return cityList


def initial_population(pop_size):
    population = []
    for i in range(0, pop_size):
        population.append(create_this())
    return population


def rank_this(population):
    fitness_results = {}
    for i in range(0, len(population)):
        fitness_results[i] = population[i].score
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


def selection(pop_ranked, elite_size):
    selection_results = []
        
    df = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(pop_ranked[i][0])
    for i in range(0, len(pop_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(0, len(pop_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(pop_ranked[i][0])
                break
    return selection_results



def mating_pool(population, selection_results):
    matingpool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        matingpool.append(list(population[index]))
    return matingpool


def breed(parent1, parent2):
    child = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)

    for i in range(0, start_gene):
        child.append(parent1[i])
    for i in range(start_gene, end_gene):
        child.append(parent2[i])
    for i in range(end_gene, len(parent1)):
        child.append(parent1[i])  
    return child


def breed_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, elite_size):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if (random.random() < mutation_rate):
            if (individual[i] == 0):
                individual[i] = 1
            else:
                individual[i] = 0
    return Genotype(individual)


def mutate_population(population, mutation_rate, elite_size):
    mutated_pop = [Genotype(fen) for fen in population[:elite_size]]

    for ind in range(elite_size, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop


def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_this(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    next_generation = mutate_population(children, mutation_rate, elite_size)
    return next_generation


def genetic_algorithm(pop_size=100, elite_size=20, mutation_rate=0.01, generations=500):     
    pop = initial_population(pop_size)
    progress = []
    for _ in progress_bar(range(generations), "Processing {} iteration"):
        pop = next_generation(pop, elite_size, mutation_rate)
        progress.append(len(sorted(pop, key=lambda x: x.score, reverse=True)[0].fenotype))


    best_index = rank_this(pop)[0][0]
    best = pop[best_index]

    plt.plot(progress)
    plt.ylabel('Number of cities')
    plt.xlabel('Generation')
    mi = math.floor(min(progress))
    ma = math.ceil(max(progress))
    plt.yticks(range(mi, ma+1, (ma-mi + 10) // 10))
    fontargs = {
        'fontsize': 'large',
        'fontweight': 'bold',
    }
    plt.text(len(progress)-1, progress[-1], str(progress[-1]), **fontargs)
    plt.text(0, progress[0], str(progress[0]), **fontargs)
    plt.savefig('progress.png')

    return best


if __name__ == "__main__":
    genetic_algorithm(pop_size=2, elite_size=1, mutation_rate=0.01, generations=10)
