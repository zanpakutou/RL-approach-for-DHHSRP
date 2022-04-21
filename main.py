from enviroment import Enviroment, Request
from schedule import Schedule
from feature_engineering import FeatureExtractor
from dqn import DQNAgent
import numpy as np
import utils
def test_result():
    env = Enviroment()
    env.make("instances/test/" + str(0) + ".in", "instances/context.in")
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
    batch_size = 4
    EPISODES = 10001
    state_size = 9
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    null_request  = Request(current_time = 0, require_time = (5, 5, 5), require_skill = 0, location = (40, 40))
    log = open("log", "w")
    np.random.seed(333)
    # agent.load("save/dhhsrp-dqn.h5")

    
    for e in range(EPISODES):
        #Init episode by random instance
        no_instance = np.random.randint(500)
        env = Enviroment()
        env.make("instances/train/" + str(no_instance) + ".in", "instances/context.in")
        sched = Schedule(env)
        requests = env.get_request()
        feature_extractor = FeatureExtractor(env, sched)
        score = 0
        verbose = e % 10 == 0
        
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
                   
                    #Calculate state
                    state = feature_extractor.get_feature(request, current_time)
                    state = np.reshape(state, [1, state_size])
                    
                    #Derive action
                    action = agent.act(state)
                    if (verbose):
                        log.write(str(state))
                        log.write(str(action))
                    #Calculate reward
                    reward = 0
                    if action == 0 or valid == False:
                        reward = 0
                    else:
                        
                        sched.accept_request(request, current_time)
                        reward = 1
                        score = score + 1
                        
                    null_request.current_time = request.current_time
                    next_state = feature_extractor.get_feature(null_request, current_time)
                    next_state = np.reshape(next_state, [1, state_size])  
                    #Check if end of episode
                    if week == env.nb_weeks - 1 and day == env.day_per_week - 1 and request == requests[week][day][-1]:
                        done = 1
                        agent.update_target_model()
                    else:
                        done = 0
                    #Experience replay
                    if valid == True or done == 1:
                        agent.memorize(state, action, reward, next_state, done)                    
                        if len(agent.memory) > batch_size:
                            agent.replay(batch_size)
                            
        print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, score, agent.epsilon))
        #Update epsilon of greedy
        if e % 5 == 0:
            print("test: {}".format(test_result()))
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
        if e % 10 == 0:
            agent.save("save/dhhsrp-dqn.h5")
        if verbose :
            log.write("episode: {}/{}, score: {}, e: {:.2}"
                            .format(e, EPISODES, score, agent.epsilon))
            
    log.close()