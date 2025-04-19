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

**Incrementally Gathering the Data**: We support incrementally collecting the samples for Figure 14. Please refer to `exp.py` in the `src/Overall Parallel/` folder.

**Figure 15**: Execute `src/Varying Transition Matrix Combination/exp.py`. The output: `result.txt` and `photo.png` will be placed in the different directories under `src/Varying Transition Matrix Combination/`, with different molecule/ion names on it.

**Figure 16**: Execute `src/Matrix Spectra/exp.py`. The graph output will be placed in the same directory.

**Figure 17**: Execute `src/Impact of Evolution Time/exp.py`. The output will be placed in the different directories under `src/Impact of Evolution Time/`, with different molecule/ion names on it. The file `result1.txt`/`photo1.png`, `result2.txt`/`photo2.png`, `result3.txt`/`photo3.png`, and `result4.txt`/`photo4.png` corresponding to evolution time $\pi/6$, $\pi/3$, $\pi/2$, and $3*\pi/4$ respectively.


**Tables2**: Execute `src/Compilation Time/exp.py`. The output: `result.txt` will be placed in the different directories under `src/Compilation Time/`, with different qubit/Pauli string number combination (`Pauli_x_y` refers to a Hamiltonian with x qubits and y Pauli strings).

## Code Structure

`exp.py` in each folder is the driver that executes the random compiler with different parameters.

`randomcompiler.py` is the implementation of MarQSim algorithm.
