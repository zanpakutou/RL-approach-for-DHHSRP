from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from enviroment import utils
from agent.feature_engineering import FeatureExtractor
from agent.actor_critic import A2CAgent

import keras.backend as K
from tensorflow.keras.utils import to_categorical

import numpy as np

instance_dir = "enviroment/instances/"

def test_result(id):
    env = PatientRequest()
    env.make(instance_dir + "test/" + str(id) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == False :
                        continue

                state = feature_extractor.get_feature(request, current_time)
                state = np.reshape(state, [1, state_size])
                action = agent.act(state)
                if action == 0:
                    continue
                else:
                    sched.accept_request(request, current_time, weekly_deadline = True)
                    ans = ans + 1
    return ans
    
if __name__ == "__main__":
    done = False

    batch_size = 16
    EPISODES = 10001    

    state_size = 22
    action_size = 2
    agent = A2CAgent(state_size, action_size)
    log = open("log", "w")
    training_log = open("train_log", "w")
    
    #agent.load("save/dhhsrp-a2c-366")
    np.random.seed(333)
    
    step = 0
    for e in range(EPISODES):
        #Init episode by random instance
        no_instance = np.random.randint(500)
        env = PatientRequest()
        env.make(instance_dir + "train/" + str(no_instance) + ".in", instance_dir + "context.in")
        sched = Schedule(env)
        requests = env.get_request()
        null_request  = Request(current_time = 0, require_time = (env.scheduling_horizon, env.day_per_week, 2), require_skill = -1, location = env.nurse_depot)
        
        feature_extractor = FeatureExtractor(env, sched)
        score = 0
        verbose = e % 5 == 0
        
        # Reset episode
        states, actions, rewards = [], [], []
        ok = False
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                    
                    #Calculate state
                    state = feature_extractor.get_feature(request, current_time)
                    np_state = np.reshape(state, [1, state_size])
                    
                    #Derive action
                    if (valid == True):
                        action = agent.act(np_state)
                    else:
                        action = 0
                        
                    if (verbose and valid == True):
                        log.write(str(list(np.around(np_state, 4))) + "\n")
                        log.write(str(agent.critic.model.predict(np_state)) + " " + str(agent.actor.model.predict(np_state)) + "\n")
                    
                    #Calculate reward
                    reward = 0
                    if action == 0 or valid == False:
                        reward = 0
                    else:
                        sched.accept_request(request, current_time, weekly_deadline = True)
                        reward = 1
                        score = score + 1
                    
                    #Check next state
                    null_request  = request
                    null_request.require_skill = -1
                    next_state = feature_extractor.get_feature(null_request, current_time)
                    next_state = np.reshape(next_state, [1, state_size]) 
                    
                    # Memorize (s, a, r) for training
                    if (valid == True):
                        #agent.upd(np_state, action, reward, next_state)
                        states.append(state)
                        actions.append(np.reshape(action, [1, 1]))
                        rewards.append(np.reshape(reward, [1, 1]))
       
        agent.update_episode(states, actions, rewards)
        print("episode: {}/{}, score: {}"
                          .format(e, EPISODES, score))
        #Update log
        if e % 5 == 0:
            print("test: {}".format(test_result(0)))
            test_res = (test_result(0) + test_result(1) + test_result(2))/3
            training_log.write(str(e) + ' ' + str(test_res) + '\n')
            training_log.flush()
            if e % 10 == 0:
                agent.save("save/dhhsrp-a2c-" + str(test_res))
            
        if verbose :
            log.write("episode: {}/{}, score: {}\n".format(e, EPISODES, score))
            log.flush()
            
    log.close()
    training_log.close()