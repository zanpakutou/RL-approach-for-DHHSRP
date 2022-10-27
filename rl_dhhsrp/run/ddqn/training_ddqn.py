import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..")))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from utils.utils import set_seed
from utils.logger import Logger
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
from config.config import Config

import numpy as np
import random
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default=".",
    help='Which folder to write logs and output, generate if not exist')
parser.add_argument('--config', type=int, default=0,
    help='Which config from config.py')
parser.add_argument('--episodes', type=int, default=10000,
    help='Number of training episodes')
parser.add_argument('--batch_size', type=int, default=512,
    help='Number of sample for each NN updating')
parser.add_argument('--discount_factor', type=float, default=0.99,
    help='Discount factor.')
parser.add_argument('--NN_size', type=int, default=256,
    help='Size of each hidden layer')
parser.add_argument('--lr', type=float, default=1e-6,
    help='Learning rate of deep Q network')
parser.add_argument('--obj', type=str, default='patient', 
    choices=['patient', 'visit'],
    help='patient: maximize number of patient. visit: maximize number of visit')
parser.add_argument('--instance_type', type=str, default='uniform',
    choices=['uniform', 'cluster', 'simplify'],
    help='Type of instance')
parser.add_argument('--arr_rate', type=int, default=360,
    choices=[150, 240, 360, 60],
    help='Arrival rate')
parser.add_argument('--transition_type', type=str, default='partial',
    choices=['partial', 'full'],
    help='Transition type')

def _reward(obj: str, request: Request):
    if args.obj == "visit":
        return request.require_time[0] * request.require_time[1]
    elif args.obj == "patient":
        return 1
    return None

def evaluate(id):
    env = PatientRequest()
    env.make(instance_dir + str(id) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans = 0

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(
                    request, current_time, weekly_deadline=True
                )
                if valid == False:
                    continue
                state = feature_extractor.get_feature(
                    request, current_time, min_cost_insertion
                )
                action = agent.act(state)
                # Warm up
                if action == 0 and week > 3:
                    continue
                else:
                    sched.accept_checked_request(
                        request, min_cost_insertion, weekly_deadline=True
                    )
                    
                    if (args.obj == 'visit'):
                        ans = ans + request.require_time[0] * request.require_time[1]
                    else:
                        ans = ans + 1
    return ans

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_folder + "/save", exist_ok=True)
    config = Config(batch_size = args.batch_size, discount_factor = args.discount_factor, num_hiddens = args.NN_size,\
                    learning_rate = args.lr, num_episodes = args.episodes, instance_type= args.instance_type, arr_rate = args.arr_rate).get_configs()[args.config]
    instance_dir = config.instances_dir
    set_seed(config.seed)
    batch_size = config.batch_size
    total_episodes = int(config.num_episodes)
    agent = DDQNAgent(config)
    logger = Logger(args.output_folder)
    replay_count = 0

    for e in range(total_episodes):
        # Init episode by random instance
        no_instance = np.random.randint(config.train_instances)
        env = PatientRequest()
        env.make(instance_dir + str(no_instance) + ".in", instance_dir + "context.in")
        
        sched = Schedule(env)
        requests = env.get_request()
        feature_extractor = FeatureExtractor(env, sched)

        score = 0
        verbose = e % config.test_frequency == 0
        logger.init_lr_log(e)
        pre_state = next_state = []
        # Simulation
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    # Checking request's feasibility
                    (valid, min_cost_insertion) = sched.check_feasible(
                        request, current_time, weekly_deadline=True
                    )
                    if valid == False:
                        continue
                    # Calculatint state
                    pre_state = next_state
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion
                    )

                    # Derive action
                    action = agent.act(state)

                    # Calculate reward
                    if action == 0 and week > 3:
                        reward = 0
                    else:
                        sched.accept_checked_request(
                            request, min_cost_insertion,
                            current_time, weekly_deadline=True,
                        )
                        reward = _reward(args.obj, request) / 100
                        score = score + _reward(args.obj, request)

                    # Calculate next state
                    next_state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion, is_post_state=True
                    )

                    # Push into the experience replay buffer
                    if week > 3:
                        # State, action, reward, state, decision transition
                        if (args.transition_type == 'full'):
                            agent.memorize(pre_state, np.random.randint(2), 0, state, True)
                        agent.memorize(state, action, reward, next_state, True)
                        if verbose and np.random.randint(5) == 0:
                            logger.write_train_log(state, agent.model.predict(state, verbose = 0))
        # Replay
        if len(agent.memory) > batch_size:
            for _ in range(config.steps_per_update):
                agent.replay(batch_size)
                replay_count = replay_count + 1
                if replay_count % config.target_update == 0:
                    agent.update_target_model()
                    replay_count = 0
        # Verbose & log
        print(
            "episode: {}/{}, score: {}, e: {:.2}".format(
                e, total_episodes, score, agent.epsilon
            )
        )
        # Evaluate agent
        if e % config.test_frequency == 0:
            test_res = 0
            for test in config.test_instances:
                test_res = test_res + evaluate(test)
            test_res = test_res / len(config.test_instances)
            print("test: {}".format(test_res), flush = True)
            logger.write_test_log(e, score)
            agent.save(
                args.output_folder + "/save/dhhsrp-ddqn-"
                + str(int(np.floor(test_res))) + ".h5"
            )

        if verbose:
            logger.write_train_log(
                "episode: {}/{}, score: {}, e: {:.2}\n".format(
                    e, total_episodes, score, agent.epsilon
                )
            )

        # Update epsilon of greedy
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        logger.flush_log()
logger.close_log()
