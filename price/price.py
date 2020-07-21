import neat
import neat.nn
import pandas as pd
try:
    import cPickle as pickle
except:
    import pickle
import sys
import os.path
from neat.visualize import draw_net, plot_stats, plot_species


INPUT = None
OUTPUT = None
CONFIG = None


def eval_fitness(genomes, config):
    for index, g in genomes:
        
        net = neat.nn.FeedForwardNetwork.create(g, config)
        
        sum_square_error = 0.0
        for input, expected in zip(INPUT, OUTPUT):
            new_input = input
            output = net.activate(new_input)
            sum_square_error += ((output[0] - expected[0]) ** 2.0) / 4.0
        
        g.fitness = 100 / (1 + sum_square_error)


def run(generations):
    pop = neat.population.Population(CONFIG)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))
    winner = pop.run(eval_fitness, generations)
    return winner, stats


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    df = pd.read_csv('train.csv', sep=',')
    OUTPUT = [(x,) for x in df['price_range']]
    df.drop(['price_range'], axis=1, inplace=True)
    df = round((df - df.min()) / (df.max() - df.min()), 5)
    INPUT = [tuple(x) for x in df.values]

    CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config')
    result = run(1000)
    winner = result[0]
    stats = result[1]
    print('\nBest genome:\n{!s}'.format(winner))
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
    with open('winner_neat_xor.pkl', 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)
    draw_net(winner_net, filename="neat_xor_winner")
    plot_stats(stats, ylog=False, view=True, filename='avg_fitness_neat.svg')
    plot_species(stats, view=True, filename='speciation_neat.svg')

    hit = 0
    total = 0
    for inputs, expected in zip(INPUT, OUTPUT):
        new_input = inputs
        output = winner_net.activate(new_input)
        total += 1
        if round(output[0]) == round(expected[0]):
            hit += 1
    print('Accuracy:', float(hit)/total)
