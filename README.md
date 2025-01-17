# DinoAI: AI-Driven Dino Game with NEAT Algorithm

**DinoAI** is a Python-based project that utilizes the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to teach AI agents how to play the classic Dino game. This project combines evolutionary algorithms with PyGame to create an interactive environment where AI learns to avoid obstacles and improve its performance over generations.

## 🌟 Features

**AI-Powered Gameplay:** 
- The NEAT algorithm evolves neural networks over generations to master the game.

**Dynamic Gameplay:**
- Includes increasing difficulty with faster speeds and randomized obstacles.

**Real-Time Visualization:**
- Displays current generation, number of AI agents alive, and scores.
- Highlights fitness and neural network decision-making.

**Customizable Configurations:**
- Fully adjustable NEAT parameters via config.txt.

**Interactive Environment:**
- Built with PyGame for smooth rendering and game logic.

## 📸 In-game Video

https://github.com/user-attachments/assets/c56ac1eb-9461-4466-bae6-7685b6e39e4b

## 🚀 Getting Started

**1. Prerequisites**  
Python 3.8 or later and Python libraries:  
- `pip install pygame neat-python`

**2. Clone the Repository**  
- `git clone https://github.com/bkrkhub/DinoAI.git`  
- `cd DinoAI`  

**3. Run the Game**  
- `python runDino.py`  

## 🛠️ Configuration

You can fine-tune the AI's behavior by editing the `config.txt` file. Key parameters include:

- **Population Size**: Number of agents per generation (`pop_size`).  
- **Fitness Threshold**: Target score the AI must reach (`fitness_threshold`).  
- **Activation Functions**: Neural network activation options (`relu`, `tanh`, `sigmoid`).  

For more details on NEAT parameters, refer to the [NEAT documentation](https://neat-python.readthedocs.io/).

## 🎮 How It Works

**Game Setup:**  
- A PyGame environment simulates the Dino game with randomized obstacles.  
- The AI agents control the Dino to avoid obstacles and earn points.  

**Training:**  
- Each generation of AI agents is evaluated based on their performance (fitness score).  
- The best-performing agents evolve to create the next generation.  

**Results:**  
- Over time, the AI improves its ability to play the game by learning from previous generations.  

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
