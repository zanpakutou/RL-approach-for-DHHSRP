from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from enviroment import utils
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent

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
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
                if valid == False :
                        continue

                state = feature_extractor.get_feature(request, current_time)
                state = np.reshape(state, [1, state_size])
                action = agent.act(state)
                if action == 0:
                    continue
                else:
                    sched.accept_request(request, current_time)
                    ans = ans + 1
    return ans
    
if __name__ == "__main__":
    done = False

    batch_size = 16
    EPISODES = 10001    

    state_size = 7
    action_size = 2
    agent = DDQNAgent(state_size, action_size)
    log = open("log", "w")
    training_log = open("train_log", "w")
    
    #agent.load("save/dhhsrp-ddqn-feed.h5")
    np.random.seed(333)
    
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
        step = 0
        
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
                    #Calculate state
                    state = feature_extractor.get_feature(request, current_time)
                    np_state = np.reshape(state, [1, state_size])
                    
                    #Derive action
                    action = agent.act(np_state)
                    
                    if (verbose and valid == True):
                        log.write(str(state) + "\n")
                        log.write(str(agent.model.predict(np_state)) + "\n")
                
                    #Calculate reward
                    reward = 0
                    if action == 0 or valid == False:
                        reward = 0
                    else:
                        sched.accept_request(request, current_time)
                        reward = 1
                        score = score + 1
                    #Check if end of episode
                    next_state = feature_extractor.get_feature(request, current_time)
                    next_state = np.reshape(next_state, [1, state_size]) 
                    
                    if week == env.nb_weeks - 1 and day == env.day_per_week - 1 and request == requests[week][day][-1]:
                        done = 1
                    else:
                        done = 0
                     
                    #Experience replay
                    if valid == True or done == 1:
                        agent.memorize(np_state, action, reward, next_state, done)
                        step = step + 1                        
                        if step % 4 == 0 and len(agent.memory) > batch_size:
                            agent.replay(batch_size)
                            
        print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, score, agent.epsilon))
                          
        #Update epsilon of greedy
        if e % 5 == 0:
            agent.update_target_model()
            print("test: {}".format(test_result(0)))
            test_res = (test_result(0) + test_result(1) + test_result(2))/3
            training_log.write(str(e) + ' ' + str(test_res) + '\n')
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
        if e % 10 == 0:
            agent.save("save/dhhsrp-ddqn.h5")
            
        if verbose :
            log.write("episode: {}/{}, score: {}, e: {:.2}\n"
                            .format(e, EPISODES, score, agent.epsilon))
            
    log.close()
    training_log.close()