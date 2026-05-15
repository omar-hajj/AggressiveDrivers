import gymnasium as gym
import highway_env
from very_aggressive import VeryAggressiveVehicle

env = gym.make("highway-v0", render_mode="human", config={"vehicles_count": 5, "vehicles_density": 0.5, "lanes_count": 3})
env.reset()
while True:
    env.step(env.action_space.sample()) #Stage 1 of traffic density


#env = gym.make("highway-v0", render_mode="human", config={"vehicles_count": 8, "vehicles_density": 1.0, "lanes_count": 3})
#env.reset()
#while True:
##stage 2 


#env = gym.make("highway-v0", render_mode="human", config={"vehicles_count": 12, "vehicles_density": 1.5, "lanes_count": 3})
##env.reset()
#while True:
#    env.step(env.action_space.sample())
#stage 3

#env = gym.make("highway-v0", render_mode="human", config={"vehicles_count": 15, "vehicles_density": 2.0, "lanes_count": 3})
#env.reset()
#while True:
#    env.step(env.action_space.sample())
#Stage 4