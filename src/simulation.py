from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

class UnityEnvironmentWrapper:
    def __init__(self, env_path):
        """
        Initialize the Unity environment wrapper with the given path.
        
        Args:
            env_path (str): Path to the Unity environment application.
        """
        self.unity_env = UnityEnvironment(env_path, no_graphics_monitor=False, no_graphics=False)
        self.env = UnityToGymWrapper(self.unity_env, uint8_visual=True, allow_multiple_obs=True)

    def reset(self):
        """
        Reset the Unity environment and return the initial observation.
        
        Returns:
            tuple: Initial observation from the environment.
        """
        return self.env.reset()

    def step(self, action):
        """
        Step the environment forward with the given action.
        
        Args:
            action (list): Action to take in the environment.
        
        Returns:
            tuple: Updated observation, reward, done flag, and additional info.
        """
        return self.env.step(action)

    def close(self):
        """
        Close the Unity environment when done.
        """
        self.env.close()
