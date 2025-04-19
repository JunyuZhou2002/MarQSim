'''
Experiment Setup Script.
This script evaluates the spectrum of various combinations of transition matrices.

Key functionalities:
- Supports the implementation of Figure 16: the Transition Matrix Spectrum experiment from the paper.

Refer to exp.py for instructions on how to execute this script.
'''

from marqsim import *
import argparse

parser = argparse.ArgumentParser(description="MarQSim Transition Matrix Spectrum Experiment Settings")
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
