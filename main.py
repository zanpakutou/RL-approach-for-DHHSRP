from enviroment import Enviroment
from schedule import Schedule
from feature_engineering import FeatureExtractor
from dqn import DQNAgent
import numpy as np
import utils
def test_result():
    env = Enviroment()
    env.make("instances/test/" + str(99) + ".in", "instances/context.in")
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
    EPISODES = 11
    state_size = 9
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    np.random.seed(333)
    #agent.load("save/dhhsrp-dqn.h5")
    
    for e in range(EPISODES):
        #Init episode by random instance
        no_instance = np.random.randint(75)
        env = Enviroment()
        env.make("instances/train/" + str(no_instance) + ".in", "instances/context.in")
        sched = Schedule(env)
        requests = env.get_request()
        feature_extractor = FeatureExtractor(env, sched)
        score = 0
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
                    if valid == False :
                        continue
                    #Calculate state
                    state = feature_extractor.get_feature(request, current_time)
                    state = np.reshape(state, [1, state_size])
                
                    #Derive action
                    action = agent.act(state)
                    #Calculate reward
                    reward = 0
                    next_state = state
                    if action == 0:
                        reward = 0
                    else:
                        sched.accept_request(request, current_time)
                        next_state = feature_extractor.get_feature(request, current_time)
                        next_state = np.reshape(next_state, [1, state_size])
                        reward = 1
                        score = score + 1
                    #Check if end of episode
                    if week == env.nb_weeks - 1 and day == env.day_per_week - 1 and request == requests[week][day][-1]:
                        done = 1
                    else:
                        done = 0
                    agent.memorize(state, action, reward, next_state, done)                    
                    
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
        print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, score, agent.epsilon))
        #Update epsilon of greedy
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if e % 10 == 0:
            agent.save("save/dhhsrp-dqn.h5")
        if e % 5 == 0:
            print("test: {}".format(test_result()))