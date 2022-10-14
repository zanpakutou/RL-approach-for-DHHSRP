class Config:
    def __init__(self):    
        self.seed = 1;
        self.instances_dir = "../../enviroment/instances/new_instances/uniform/150/"
        self.train_instances = 500
        self.test_instances = [501, 502, 503]
        #training parameter
        self.batch_size = 512;
        self.num_episodes = 100000;
        self.test_frequency = 20;
        self.steps_per_update = 10;
        self.target_update = 200;
        #agent's parameter
        self.memory_size = 500
        self.discount_factor = 0.99;
        self.epsilon_decay = 0.9995;
        self.epsilon_max = 1;
        self.epsilon_min = 0.1;
        self.learning_rate = 0.000001;
        self.num_hiddens = 256;
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
        self.instances_dir = "../../enviroment/instances/simplify/240/"
        #training parameter
        self.num_hiddens = 64;
        self.state_size = 9
        self.action_size = 2
        return self
    def get_config_3(self):
        self.__init__()
        self.instances_dir = "../../enviroment/instances/simplify/240/"
        self.memory_size = 200000;
        self.steps_per_update = 20;
        self.target_update = 400
        #training parameter
        self.num_hiddens = 64;
        self.state_size = 9
        self.action_size = 2
        return self
    def get_configs(self):
        return [Config(), Config().get_config_1(), Config().get_config_2(), Config().get_config_3()]