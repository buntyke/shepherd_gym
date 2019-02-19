##tests the performance of the learning model as a function of the number of sheep. assume the following constant: number of trained episodes at 1000. 
import argparse

from subprocess import call
from statistics import mean

def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description='Script for preprocess')
    parser.add_argument('-r', '--repetitions', type=int, default=5,
                        help='number of repetitions')
    parser.add_argument('--number_of_episodes', type=int, default=25,
                        help='number of episodes')
    parser.add_argument('--num_sheep', type=int, default=5,
                        help='number of sheep')

    
    
    # parse arguments
    args = parser.parse_args()
    repetitions = args.repetitions
  #  start_trained_demos = args.start_trained_demos
  #  end_trained_demos = args.end_trained_demos
  #  trained_demos_step = args.trained_demos_step
    number_of_episodes = args.number_of_episodes
    num_sheep = args.num_sheep
    
    success_rates = []
    
    for sheep in range(int(num_sheep), 150, 1):
        call(['rm', 'temp_results.txt'])
        
        for i in range(0, repetitions):
            print("TESTING NOW: {}".format(sheep))
            call(['python', 'dataset_process.py', '-d', '../data/heuristic'])
            call(['python', 'shepherd_imitation.py', '-e', 'heuristic', '--num_sheep', '30', '--nocuda'])
            call(['python', 'shepherd_imitation.py', '-e', 'heuristic', '-m', 'test', '-n', '20', '--num_sheep',  str(sheep),'--nocuda'])
        

        # Show the results from the file and calculate
        results = []
        with open("temp_results.txt", "r") as result_file:
            for line in result_file:
                results.append(float(line.strip()))

        print("Mean Success Rate: {}".format(mean(results)))
        
        current_success_rate = {
            'num_sheep' : sheep,
            'mean_success_rate' : mean(results),
            'success_rates' : results,
        }
        success_rates.append(current_success_rate)
    
    for sr in success_rates:
        print("num_sheep: {}, success_rates: {}, mean_success_rate: {}".format(sr['num_sheep'],
                                                                               sr['success_rates'],
                                                                               sr['mean_success_rate']))
        

if __name__ == "__main__":
    main()
    

