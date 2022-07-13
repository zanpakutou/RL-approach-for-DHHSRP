from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from enviroment import utils
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
import keras.backend as K

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
                mask = [0]
                for nurse in range(nb_nurse):
                    check = sched.check_feasible(request, current_time, spec_nurse = nurse, weekly_deadline = True)
                    if (check[0] == True):
                        mask.append(nurse + 1)
                #Derive action
                action = agent.act(np_state, mask)
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
    nb_nurse = 6
    
    state_size = 27
    action_size = 1 + nb_nurse
    
    agent = DDQNAgent(state_size, action_size)
    log = open("log", "w")
    training_log = open("train_log", "w")
    
    #agent.load("save/start.h5")
    np.random.seed(333)
    
    step = 0
    for e in range(EPISODES):
        #Init episode by random instance
        no_instance = np.random.randint(500)
        env = PatientRequest()
        env.make(instance_dir + "train/" + str(no_instance) + ".in", instance_dir + "context.in")
        sched = Schedule(env)
        requests = env.get_request()
        null_request  = Request(current_time = 0, require_time = (env.scheduling_horizon, env.day_per_week, 2), require_skill = 0, location = env.nurse_depot)
        
        feature_extractor = FeatureExtractor(env, sched)
        score = 0
        verbose = e % 10 == 0
        
        
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                    #Calculate state
                    state = feature_extractor.get_feature(request, current_time)
                    np_state = np.reshape(state, [1, state_size])
                    #Create mask of actions
                    mask = [0]
                    for nurse in range(nb_nurse):
                        check = sched.check_feasible(request, current_time, spec_nurse = nurse, weekly_deadline = True)
                        if (check[0] == True):
                            mask.append(nurse + 1)
                    #Derive action
                    action = agent.act(np_state, mask)
                    
                    if (verbose and valid == True):
                        log.write(str(state) + "\n")
                        log.write(str(agent.model.predict(np_state)) + "\n")
                
                    #Calculate reward
                    reward = 0
                    if action == 0 or valid == False:
                        reward = 0
                    else:
                        sched.accept_request(request, current_time, spec_nurse = action - 1, weekly_deadline = True)
                        reward = 1
                        score = score + 1
                    #Check if end of episode
                    next_state = feature_extractor.get_feature(request, current_time)
                    next_state = np.reshape(next_state, [1, state_size])
                     
                    #Experience replay
                    if valid == True or done == 1:
                        agent.memorize(np_state, action, reward, next_state, done)
                        step = step + 1                        
                        if step % 4 == 0 and len(agent.memory) > batch_size:
                            agent.replay(batch_size)
                        if step % 1000 == 0:
                            agent.update_target_model()
                            
        print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, score, agent.epsilon))
                          
        #Update epsilon of greedy
        if e % 5 == 0:
            print("test: {}".format(test_result(0)))
            test_res = (test_result(0) + test_result(1) + test_result(2))/3
            training_log.write(str(e) + ' ' + str(test_res) + '\n')
            training_log.flush()
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            if e % 10 == 0:
                agent.save("save/dhhsrp-ddqn-" + str(test_res) + ".h5")
            
        if verbose :
            log.write("episode: {}/{}, score: {}, e: {:.2}\n"
                            .format(e, EPISODES, score, agent.epsilon))
            log.flush()
            
    log.close()
    training_log.close()