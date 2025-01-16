# Quantum-Annealing-Enhanced RRT Algorithm for Robotic Path Planning in Mine Detection and Clearance

## Overview
This project implements a hybrid approach combining the Rapidly-Exploring Random Tree (RRT) algorithm with quantum annealing for robotic path planning in mine detection and clearance. The goal is to develop an efficient and safe navigation system for robots to detect and clear landmines, addressing a critical global humanitarian issue. The project leverages classical RRT* for initial path planning and quantum annealing for global optimization, aiming to improve path quality and computational efficiency.

---

## Repository Structure
The repository contains the following files:

1. **README.md**: This file provides an overview of the project, instructions for running the code, and examples of the output.
2. **Classic Code (1).ipynb**: Jupyter Notebook containing the implementation of the classical RRT* algorithm for robotic path planning.
3. **Quantum code.py**: Python script integrating quantum annealing using Qiskit for optimizing the RRT* algorithm.
4. **image.png**: Example visualization of the path planning process.

---

## How to Run the Codebase

### Note:
The quantum code should run on spyder IDE

### Prerequisites
1. Install Python 3.8 or later.
2. Install the required libraries using the following commands:

```bash
# Install Pygame for visualization
conda install pygame

# Install Qiskit libraries for quantum annealing
pip install qiskit-optimization
pip install qiskit
pip install qiskit-algorithms qiskit-terra qiskit-primitives

```

## Running the Code
The repository contains the following files:

1. **README.md**: This file provides an overview of the project, instructions for running the code, and examples of the output.
2. **Classic Code (1).ipynb**: Jupyter Notebook containing the implementation of the classical RRT* algorithm for robotic path planning.
3. **Quantum code.py**: Python script integrating quantum annealing using Qiskit for optimizing the RRT* algorithm.
4. **classical code output.jpeg**: Example visualization of the path planning process using the classical code.
5. **quantum code output.jpeg**: Example visualization of the path planning process using the quantum code.

## Example outputs

###Example 1: Classical RRT* 
Input: Start = (10, 10), Goal = (400, 400), Obstacles = [ ( 100 , 100 ) , ( 200 , 300 ) , ( 350 , 150 ) ] [(100,100),(200,300),(350,150)] Output: A collision-free path from the start to the goal, visualized using Pygame.
Visualization: shown in classical code output.jpeg

###Example 2: Quantum-Enhanced RRT* 
Input: Same as Example 1. 
Output: A globally optimized path with reduced cost, leveraging quantum annealing.
Visualization: shown in quantum code output.jpeg

