import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from utils.utils import train_log, test_log, lr_log
from utils.utils import set_seed, open_log, flush_log, close_log
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
from config.config import Config

import numpy as np
import random

config = Config().get_config_1()
instance_dir = config.instances_dir

def evaluate(id):
    env = PatientRequest()
    env.make(instance_dir + str(id) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans = 0
    saved_epsilon = agent.epsilon
    agent.epsilon = 0
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
                state = np.reshape(state, [1, state_size])
                action = agent.act(state)
                #Warm up   
                if week <= 3:
                    action = 1
                if action == 0:
                    continue
                else:
                    sched.accept_checked_request(
                        request, min_cost_insertion, current_time, weekly_deadline=True
                    )
                    ans = ans + 1
    agent.epsilon = saved_epsilon
    return ans

if __name__ == "__main__":
    set_seed(config.seed)
    #open_log()
    batch_size = config.batch_size
    EPISODES = 100001

    state_size = config.state_size
    action_size = config.action_size
    agent = DDQNAgent(config)

    replay_count = 0
    for e in range(EPISODES):
        # Init episode by random instance
        no_instance = np.random.randint(config.train_instances)
        env = PatientRequest()
        env.make(instance_dir + str(no_instance) + ".in", instance_dir + "context.in")
        sched = Schedule(env)
        requests = env.get_request()

        feature_extractor = FeatureExtractor(env, sched)
        score = 0
        verbose = e % config.test_frequency == 0
        lr_log.write("episodes : {}\n".format(e))
        prev_state = []
        next_state = []

        
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(
                        request, current_time, weekly_deadline=True
                    )
                    if (valid == False):
                        continue
                    # Calculate state
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion
                    )
                    state = np.reshape(state, [1, state_size])
                    prev_state = next_state
                    # Derive action
                    action = agent.act(state)
                    if verbose and np.random.randint(5) == 0:
                        pred = agent.model.predict(state)
                        train_log.write(str(np.around(state, 4).tolist()) + "\n")
                        train_log.write(str(np.around(pred, decimals=4)) + "\n")
                    #Warm up   
                    if week <= 3:
                        action = 1
                    # Calculate reward
                    reward = 0
                    if action == 0:
                        reward = 0
                    else:
                        sched.accept_checked_request(
                            request, min_cost_insertion,
                            current_time, weekly_deadline=True,
                        )
                        reward = 0.01
                        score = score + 1
                    # Check if end of episode
                    next_state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion, is_post_state=True
                    )
                    next_state = np.reshape(next_state, [1, state_size])
                    
                    # Push into the experience replay buffer
                    if week > 3:
                        #State, action, reward, state, decision transition
                        agent.memorize(state, action, reward, next_state, True)
        #Replay
        if len(agent.memory) > batch_size:
            for _ in range(config.steps_per_update):
                agent.replay(batch_size)
                replay_count = replay_count + 1
                if replay_count % config.target_update == 0:
                    agent.update_target_model()
                    replay_count = 0
        #Verbose & log
        print(
            "episode: {}/{}, score: {}, e: {:.2}".format(
                e, EPISODES, score, agent.epsilon
            )
        )
        if e % config.test_frequency == 0:
            test_res = 0
            for test in config.test_instances:
                test_res = test_res + evaluate(test)
            test_res = test_res / len(config.test_instances)
            print("test: {}".format(test_res))
            test_log.write(str(e) + " " + str(test_res) + "\n")
            
            agent.save("save/dhhsrp-ddqn-" + str(test_res) + ".h5")
        if verbose:
            train_log.write(
                "episode: {}/{}, score: {}, e: {:.2}\n".format(
                    e, EPISODES, score, agent.epsilon
                )
            )
        flush_log()
        # Update epsilon of greedy
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    close_log()