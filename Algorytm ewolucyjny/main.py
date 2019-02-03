#!/usr/bin/env python3
"""
This is entrypoint of hospital-allocating project
"""
import sys
from hospitals import gmaps
import argparse
from hospitals.geneticAlgorithm import genetic_algorithm


def _parse_cli(defaults):
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-i', '--iterations', '--iters', type=int, default=defaults['iterations'],
                            help='Number of iterations to perform.\n'
                                 ' [DEFAULT {}]'.format(defaults['iterations']))

    arg_parser.add_argument('-s', '--specimen', nargs=2, type=int, default=defaults['specimen'],
                            help='Number of specimen for N->M algorithm.'
                                 '[DEFAULT {} {}]'.format(*defaults['specimen']))

    arg_parser.add_argument('-r', '--mutationrate', type=int, default=defaults['mutationrate'],
                            help='Mutation rate.\n'
                                 ' [DEFAULT {}]'.format(defaults['mutationrate']))

    return arg_parser.parse_args()


def parser():
    """Take care of parsing arguments from CLI."""

    defaults = {
        'iterations': 100,
        'specimen': (20, 80),
        'mutationrate': 0.01,
    }

    cli_args = _parse_cli(defaults)

    if cli_args.iterations <= 0:
        raise argparse.ArgumentTypeError('{} iterations don\'t make sense!'.format(cli_args.iterations))

    if any(i <= 0 for i in cli_args.specimen):
        raise argparse.ArgumentTypeError('Values: {} for specimen count don\'t make sense!'.format(cli_args.specimen))

    if not 0 <= cli_args.mutationrate <= 1:
        raise argparse.ArgumentTypeError('{} mutations doesn\'t make sense!'.format(cli_args.mutationrate))

    return vars(cli_args).copy()


def main():
    try:
        args = parser()
    except argparse.ArgumentTypeError as e:
        print(e)
        return 1
    elite_size = args['specimen'][0]
    population_size = elite_size + args['specimen'][1]
    generations = args['iterations']
    mutation_rate = args['mutationrate']

    m = gmaps.Map()

    fenotype = genetic_algorithm(pop_size=population_size, elite_size=elite_size,
                                 mutation_rate=mutation_rate, generations=generations).fenotype

    for shape in fenotype.shapes:
        m.polygon(*zip(*shape), color='green', face_alpha=0.2)

    for shape in fenotype._shape():
        m.polygon(*zip(*shape), color='blue', face_alpha=0.1)

    m.draw('gmaps_c.html')

    return 0


if __name__ == '__main__':
    sys.exit(main())
