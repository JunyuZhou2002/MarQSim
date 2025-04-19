# MarQSim: Reconciling Determinism and Randomness in Compiler Optimization for Quantum Simulation

Artifact for paper _MarQSim: Reconciling Determinism and Randomness in Compiler Optimization for Quantum Simulation_

---

## System Requirement

**Hardware**:

* 40GB+ available hard drive
* 32GB+ RAM
* CUDA GPU (highly recommended for faster evaluation)

**Software**:

* Linux (Ubuntu 22.04 server is tested)
* Python 3.10+

## Setup

We will use Python virtual environment. To set up and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install necessary Python packages:

```bash
pip3 install -r requirements.txt
```

You can now use your favorite frontend to execute the code (vscode, browser, etc.)

## Results

This section shows how to obtain result for each figure/table from the MarQSim paper.

**Figure 14**: Execute `python3 exp.py --experiment=overall` under folder `src`. The output: `result.txt` and `result.png` will be placed in the different directories under `src/Overall_Improvement/`, with different molecule/ion names on it. A table is provided to illustrate the amount of execution time used to generate the results on our machine.

| Na+   | Cl-   | Ar    | OH-   | HF    | LiH(f) | BeH2(f) | LiH    | H2O    | SYK 1 | SYK 2 | BeH2   |
|-------|-------|-------|-------|-------|--------|---------|--------|--------|-------|-------|--------|
| 0.8 h | 1 h   | 1.65 h | 5.1 h | 24.7 h | 1.3 h  | 7 day  | 2.6 day | 10 day | 1.24  h | 3.44 h | 3.5 day |

**Incrementally Gathering the Data**: We support incrementally collecting the samples for Figure 14. Please refer to `exp.py`.

**Figure 15**: Execute `python3 exp.py --experiment=varying` under folder `src`. The output: `result.txt` and `result.png` will be placed in the different directories under `src/Varying_Combination/`, with different molecule/ion names on it.

**Figure 16**: Execute `python3 exp.py --experiment=spectra` under folder `src`. The graph output will be placed under `src/Matrix_Spectra/`.

**Figure 17**: Execute `python3 exp.py --experiment=evol` under folder `src`. The output will be placed in the different directories under `src/Evolution_Time_Impact/`, with different molecule/ion names on it. The end of filename `result***.txt`/`result***.png` will indicate the corresponding to evolution time $\pi/6$, $\pi/3$, $\pi/2$, and $3*\pi/4$ respectively.

**Tables2**: Execute `python3 exp.py --experiment=time` under folder `src`. The output: `result.txt` will be placed in the different directories under `src/Compilation_Time/`, with different qubit/Pauli string number combination (`Pauli_x_y` refers to a Hamiltonian with x qubits and y Pauli strings).

**Note**: You can also use each command string from the experiment list in `exp.py` and execute them individually. This approach also displays a process indicator that reflects the progress of the sampling process.

## Code Structure

1. `marqsim.py` contains the core functions for implementing the MarQSim compiler as proposed in the paper.
2. `overall.py`, `overall_parallel.py`, `varying.py`, `spectra.py`, and `compile_time.py` handle sampling and data processing to reproduce the results from our paper, leveraging functionalities from `marqsim.py`.
3. `exp.py` is a script designed to schedule and run experiments. By parsing different arguments, it allows reproduction of various experiments from the paper.
4. The Benchmarks folder includes all molecular and ion data used in the experiments. Other folders are empty and used for data collection and result visualization.


## Simulating New Molecule/Ion
To simulate new elements, prepare their Hamiltonian information following the examples in the Benchmarks folder. Then, invoke the operation function in `marqsim.py` with your desired parameters (e.g., evolution time, precision, transition matrix combination ratio, etc.).
