from unityagents import UnityEnvironment


class BananaEnv:
    """
    Banana enviroment (a wrapper around unity env) to have similar interface to openAi gym env.
    """
    def __init__(self, file_name, train_mode) -> None:
        """
        """
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        self.train_mode = train_mode
        brain = self.env.brains[self.brain_name]
        self.nA = brain.vector_action_space_size
        state = self.reset()
        self.nS = len(state)
        
    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        return state        
    
    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        return next_state, reward, done, env_info
    
    def close(self):
        self.env.close()
        



    