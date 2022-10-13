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

config = Config().get_config_2()
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
                action = agent.act(state)
                # Warm up
                if action == 0 and week > 3:
                    continue
                else:
                    sched.accept_checked_request(
                        request, min_cost_insertion, weekly_deadline=True
                    )
                    ans = ans + 1
    agent.epsilon = saved_epsilon
    return ans


if __name__ == "__main__":

    set_seed(config.seed)
    batch_size = config.batch_size
    total_episodes = config.num_episodes
    state_size = config.state_size
    action_size = config.action_size
    agent = DDQNAgent(config)
    logger = Logger()

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
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion
                    )

                    # Derive action
                    action = agent.act(state)

                    # Calculate reward
                    if action == 0 and week > 3:
                        reward = -0.005
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
                        # State, action, reward, state, decision transition
                        agent.memorize(state, action, reward, next_state, True)
                        if verbose and np.random.randint(5) == 0:
                            logger.write_train_log(state, agent.model.predict(state))
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
            print("test: {}".format(test_res))
            logger.write_test_log(e, score)
            agent.save(
                "save/dhhsrp-ddqn-"
                + str(int(np.floor(test_res))) + "-"
                + str(agent.epsilon)   + ".h5"
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
