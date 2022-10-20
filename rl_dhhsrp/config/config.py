class Config:
    def __init__(self, batch_size = 512, num_episodes = 1e5, discount_factor = 0.99, learning_rate = 1e-6, num_hiddens = 512):    
        self.seed = 1;
        self.instances_dir = "../../enviroment/instances/new_instances/uniform/150/"
        self.train_instances = 500
        self.test_instances = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510]
        #training parameter
        self.batch_size = batch_size;
        self.num_episodes = num_episodes;
        self.test_frequency = 100;
        self.steps_per_update = 10;
        self.target_update = 200;
        #agent's parameter
        self.memory_size = 1000000
        self.discount_factor = discount_factor;
        self.epsilon_decay = 0.9998;
        self.epsilon_max = 1;
        self.epsilon_min = 0.1;
        self.learning_rate = learning_rate;
        self.num_hiddens = num_hiddens;
        self.num_layers = 2;
        self.state_size = 29
        self.action_size = 2
    def get_config_1(self):
        self.__init__()
        #training parameter
        self.instances_dir = "../../enviroment/instances/new_instances/uniform/240/"
        #agent's parameter
        return self
    def get_config_2(self):
        self.__init__()
        #training parameter
        self.instances_dir = "../../enviroment/instances/new_instances/uniform/360/"
        #agent's parameter

        return self
    def get_config_3(self):
        self.__init__()
        self.instances_dir = "../../enviroment/instances/simplify/240/"
        self.learning_rate = 0.000003;
        self.state_size = 9
        self.action_size = 2
        self.num_hiddens = 128;

        return self
    def get_configs(self):
        return [self, self.get_config_1(), self.get_config_2(), self.get_config_3()]