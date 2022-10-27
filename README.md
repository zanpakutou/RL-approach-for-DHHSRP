# RL-approach-for-DHHSRP
Reinforcement learning approach for Dynamic Home Health Care Scheduling and Routing Problem
## Structure

    .
    ├── agent                   <- RL agent
    ├── enviroment             	<- Instances and scheduling classes
    ├── greedy                  <- Code for greedy algorithm(s)
    ├── utils                    
    └── run			<- Code for DDQN, SB, test...
## Setup
### Install enviroment
```cd 
python3.8 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```
### Training
```
python training.py
```
### Running
```
python evaluate.py
```
