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

We will use Python virtual environment together with Jupyter notebook. To set up and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install necessary Python packages:

```bash
pip3 install -r requirements.txt
```

You can now use your favorite Jupyter notebook frontend to execute the code (vscode, browser, etc.)

## Results

Layout may be different from the original paper. Slight variation is expected as the simulation and MarQSim itself introduces randomness. The following section shows how to obtain result for each figure/table from the paper.

**Figure 14**: Execute `src/Overall Improvement/exp.py`. The output: `result.txt` and `photo.png` will be placed in the different directories under `src/Overall Improvement/`, with different molecule/ion names on it. Usually molecule/ion with 8 qubit take hours to compile while molecule/ion with 12/14 qubit can take days (molecule/ion size can be seen in `Table 1`).

**Figure 15**: Execute `src/Varying Transition Matrix Combination/exp.py`. The output will be placed in the different directories under `src/Varying Transition Matrix Combination/`, with different molecule/ion names on it.

**Figure 16**: Execute `src/Matrix Spectra/exp.py`. The graph output will be placed in the same directory.

**Figure 17**: Execute `src/Impact of Evolution Time/exp.py`. The output will be placed in the different directories under `src/Impact of Evolution Time/`, with different molecule/ion names on it. The file `result1.txt`/`photo1.png`, `result2.txt`/`photo2.png`, `result3.txt`/`photo3.png`, and `result4.txt`/`photo4.png` corresponding to evolution time $\pi/6$, $\pi/3$, $\pi/2$, and $3*\pi/4$ respectively.


**Tables**:
