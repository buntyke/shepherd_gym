#tests the performance of the learning model as a function of the number of trained episodes. assume constant: number of sheep. 

import argparse

from subprocess import call
from statistics import mean

def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description='Script for preprocess')
    parser.add_argument('-r', '--repetitions', type=int, default=5,
                        help='number of repetitions')
    parser.add_argument('--start_trained_demos', type=int, default=560,
                        help='number of trained demos at the start')
    parser.add_argument('--end_trained_demos', type=int, default=941,
                        help='number of trained demos at the end')
    parser.add_argument('--trained_demos_step', type=int, default=20,
                        help='steps between trained demos')
    parser.add_argument('--number_of_episodes', type=int, default=25,
                        help='number of episodes')
    parser.add_argument('--num_sheep', type=int, default=25,
                        help='number of sheep')


    
    
    # parse arguments
    args = parser.parse_args()
    repetitions = args.repetitions
    start_trained_demos = args.start_trained_demos
    end_trained_demos = args.end_trained_demos
    trained_demos_step = args.trained_demos_step
    number_of_episodes = args.number_of_episodes
    num_sheep = args.num_sheep
    
    success_rates = []
    for trained_demos in range(int(start_trained_demos), int(end_trained_demos), int(trained_demos_step)):
        # Delete the old results file
        call(['rm', 'temp_results.txt'])

        #call(['python', 'dataset_process.py', '-d', '../data/heuristic', '-n', str(trained_demos), '-p', '0.1'])
        #call(['python', 'dataset_process.py', '-d', '../data/heuristic', '-n', str(trained_demos)])

        for i in range(0, repetitions):
            call(['python', 'dataset_process.py', '-d', '../data/heuristic', '-n', str(trained_demos)])
            call(['python', 'shepherd_imitation.py', '-e', 'heuristic', '--nocuda'])
            call(['python', 'shepherd_imitation.py', '-e', 'heuristic', '-m', 'test', '-n', str(number_of_episodes),'--nocuda'])
        
        #call(['python', 'shepherd_imitation.py', '-e', 'heuristic', '-m', 'test', '-n', str(number_of_episodes),'--nocuda', '--num_sheep', str(num_sheep)])
        


        # Show the results from the file and calculate
        results = []
        with open("temp_results.txt", "r") as result_file:
            for line in result_file:
                results.append(float(line.strip()))

        print("Mean Success Rate: {}".format(mean(results)))
        
        current_success_rate = {
            'trained_demos' : trained_demos,
            'mean_success_rate' : mean(results),
            'success_rates' : results,
        }
        success_rates.append(current_success_rate)
    
    for sr in success_rates:
        print("trained_demos: {}, success_rates: {}, mean_success_rate: {}".format(sr['trained_demos'],
                                                                                   sr['success_rates'],
                                                                                   sr['mean_success_rate']))
        

if __name__ == "__main__":
    main()
    

