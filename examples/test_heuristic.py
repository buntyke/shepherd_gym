#tests performance of the heuristic model as a function of the number sheep. no learning involved whatsoever. 
import argparse

from subprocess import call
from statistics import mean

def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description='Script for preprocess')
    parser.add_argument('--num_sheep', type=int, default=5,
                        help='number of sheep')


    
    
    # parse arguments
    args = parser.parse_args()
    num_sheep = args.num_sheep
    
    success_rates = []
    
    for sheep in range(int(num_sheep), 151, 1):
        call(['python', 'shepherd_heuristic.py', '-e', 'heuristic', '-s', '55', '-n', '25', '--norender', '--noplot', '--num_sheep', str(sheep), '--evaluate'])

        # Show the results from the file and calculate
        

        print("Success Rate: {}".format(success_rates))
        
        current_success_rate = {
            'num_sheep' : sheep,
            'success_rates' : success_rates,
        }
        success_rates.append(current_success_rate)
    
    for sr in success_rates:
        print("num_sheep: {}, success_rates: {}".format(sr['num_sheep'], sr['success_rates']))
        

if __name__ == "__main__":
    main()
    

