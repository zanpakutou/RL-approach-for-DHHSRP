import copy 

class Config:
    def __init__(self, config = None, batch_size = 512, num_episodes = 1e5, discount_factor = 0.99, learning_rate = 1e-6, num_hiddens = 512, instance_type='uniform', arr_rate=360):    
        self.seed = 1;
        self.instances_dir = '../../enviroment/instances/' + instance_type +  '/' + str(arr_rate) +'/'
        self.train_instances = 500
        self.test_instances = [501, 502, 503, 504, 505]
        #training parameter
        self.batch_size = batch_size;
        self.num_episodes = num_episodes;
        self.test_frequency = 100;
        self.steps_per_update = 20;
        self.target_update = 400;
        #agent's parameter
        self.memory_size = 1000000
        self.discount_factor = discount_factor;
        self.epsilon_decay = 0.9998;
        self.epsilon_max = 1;
        self.epsilon_min = 0.05;
        self.learning_rate = learning_rate;
        self.num_hiddens = num_hiddens;
        self.num_layers = 2;
        self.state_size = 30
        if (instance_type == 'simplify'):
            self.state_size = 10
        self.action_size = 2
        
        if (config != None):
            self = copy.deepcopy(config)
        
    def get_config_1(self):
        new_config = Config(self)
        new_config.instances_dir = "../../enviroment/instances/new_instances/uniform/240/"
        return new_config

    def get_config_2(self):
        new_config = Config(self)
        new_config.instances_dir = "../../enviroment/instances/new_instances/uniform/360/"
        return new_config

    def get_config_3(self):
        new_config = Config(self)
        new_config.instances_dir = "../../enviroment/instances/simplify/240/"
        new_config.state_size = 10
        return new_config
        
    def get_configs(self):
        return [self, self.get_config_1(), self.get_config_2(), self.get_config_3()]