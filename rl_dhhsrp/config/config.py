class Config:
    def __init__(self):    
        self.seed = 1;
        self.instances_dir = "../../enviroment/instances/new_instances/uniform/240/"
        self.train_instances = 500
        self.test_instances = [501, 502, 503]
        #training parameter
        self.batch_size = 64;
        self.num_episodes = 20000;
        self.test_frequency = 10;
        self.steps_per_update = 250;
        self.target_update = 750;
        #agent's parameter
        self.memory_size = 100000;
        self.discount_factor = 0.99
        self.epsilon_decay = 0.9995;
        self.epsilon_max = 1;
        self.epsilon_min = 0.1;
        self.learning_rate = 0.0003;
        self.num_hiddens = 512;
        self.num_layers = 1;
        self.state_size = 16
        self.action_size = 2
    def get_config_1(self):
        self.__init__()
        #training parameter
        self.batch_size = 64;
        self.steps_per_update = 250;
        self.target_update = 1000;
        #agent's parameter
        self.memory_size = 200000;
        self.epsilon_decay = 0.9993;
        self.learning_rate = 0.0005;
        self.num_hiddens = 512;
        self.num_layers = 1;
        return self
    def get_config_2(self):
        self.__init__()
        #training parameter
        self.batch_size = 64;
        self.steps_per_update = 200;
        self.target_update = 500;
        #agent's parameter
        self.memory_size = 500000;
        self.epsilon_decay = 0.9995;
        self.learning_rate = 0.0003;
        self.num_hiddens = 512;
        self.num_layers = 1;
        return self
    def get_config_3(self):
        self.__init__()
        #training parameter
        self.batch_size = 32;
        self.steps_per_update = 250;
        self.target_update = 500;
        #agent's parameter
        self.memory_size = 500000;
        self.epsilon_decay = 0.9993;
        self.optimizer = 0.00075;
        self.num_hiddens = 16;
        self.num_layers = 2;
        return self