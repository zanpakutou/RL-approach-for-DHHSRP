# RL-approach-for-DHHSRP
Reinforcement learning approach for Dynamic Home Health Care Scheduling and Routing Problem
## Structure

    .
    ├── agent                   <- RL agent
    ├── enviroment
    │   ├── test                
    │   ├── train               
    │   └── unit                
    ├── greedy                  <- Code for greedy algorithm(s)
    ├── save                    <- Pretrain model
    ├── training.py
    └── evaluate.py
## Setup
### Install enviroment
```
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