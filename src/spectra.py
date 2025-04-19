'''
Experiment Set Up Script. 
It evaluates gate usage (CNOT and single-qubit), fidelity against exact evolution, and analyzes compilation performance across various sampling strategies and parameter settings.

Key functionalities include:
- 1. Support to implement Fig. 14, the Overall Improvement experiment in the paper.
- 2. Support to implement Fig. 17, the Impact of Evolution Time experiment in the paper.

See exp.py on how to execute this file for different experiments.
'''

from marqsim import *
import argparse

parser = argparse.ArgumentParser(description="MarQSim Compiler Experiment Settings")
# Path to the experiment folder. 
parser.add_argument('--exp_path', help='Name of the experiment. Used to determine the output folder for saving results.')
# Path to the Pauli string input file 
parser.add_argument('--file', help='Path to the Pauli string input file.')

args = parser.parse_args()

spectra_operation(file=args.file, path=args.exp_path)

# commend line
'''
python3 spectra.py --exp_path=Matrix_Spectra --file=Pauli_Na+
'''
