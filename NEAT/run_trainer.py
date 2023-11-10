import pickle
import sys
import numpy as np
import time
import atexit
import neat
import visualize

# MLAGENTS stuff
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

save_nn_destination = 'result/best.pkl'

# [PARAMETERS]
max_generations = 2  # Max number of generations

# Best
best_current_gen = None

single_agent_env_path = "./Builds/3DBallBalancing.exe"

no_graphics = False

env = UnityEnvironment(file_name=single_agent_env_path, seed=0, no_graphics = no_graphics)

# Reset the enviroment to get it ready  
print("ENV Has been reset")
env.reset()

# engine_configuration_channel = EngineConfigurationChannel()
# engine_configuration_channel.set_configuration_parameters(time_scale=100, width=256, height=256, target_frame_rate=-1, quality_level=0)
# env.side_channels[2] = engine_configuration_channel

behavior_specs = list(env.behavior_specs)
behavior_name = behavior_specs[0]
print(behavior_specs)
print(behavior_name)        
generation = 0


def exit_handler():
    visualize.plot_stats(stats, view=True, filename="NEAT/result/in_progress/recurrent-fitness"+str(generation)+".svg", label="CTRNN")
    visualize.plot_species(stats, view=True, filename="NEAT/result/in_progress/recurrent-speciation"+str(generation)+".svg", label="CTRNN")
    with open(save_nn_destination, 'wb') as w:
        pickle.dump(best_current_gen, w)
    print("EXITING")
    env.close

def run_agent(genomes, cfg):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    agent_to_local_map = {}
    local_to_agent_map = {}
    id_count = 0
    for step in decision_steps:
        print(step)
        agent_to_local_map[step] = id_count
        local_to_agent_map[id_count] = step
        id_count += 1
    policies = []
    for _, g in genomes:
        policy = neat.nn.RecurrentNetwork.create(g, cfg)
        policies.append(policy)
        g.fitness = 0
    print("Genomes: " + str(len(genomes)))
    global generation
    generation += 1
    done = False  # For the tracked_agent
    total_reward = 0.0
    agent_count = len(decision_steps.agent_id)
    print("Agent count: ", agent_count)
    while not done:
        actions = np.zeros(shape=(agent_count, 2))
        # Concatenate all the observation data BESIDES obs number 3 (OtherAgentsData)
        nn_input = np.zeros(shape=(agent_count, 4))  
        for agent in range(agent_count):  # Collect observations from the agents requesting input
            if local_to_agent_map[agent] in decision_steps:
                decision_steps = decision_steps
            else:
                continue
            step = decision_steps[local_to_agent_map[agent]]
            # print("Step obs: ", len(step.obs[0]))
            nn_input[agent] = step.obs[0]

        start = time.time()
        # Fetches actions by feed forward pass through the NNs
        if (len(decision_steps) > 0) and (len(decision_steps) > 0):  # More steps to take?
            for agent in range(agent_count):  # Iterates through all the agent indexes
                if (local_to_agent_map[agent] in decision_steps):  # Is agent ready?
                    # If fixed opponent, purple is controlled by fixed policy
                    if (local_to_agent_map[agent] in decision_steps):
                        action = policies[agent].activate(nn_input[agent])  # FPass for purple and blue
                    actions[agent] = action  # Save action in array of actions
                    
        end = time.time()
        time_spent_activating = (end - start)
        if len(decision_steps) != 0:
            for agent in range(agent_count):
                if (local_to_agent_map[agent] in decision_steps)    :
                    continuous_actions = [actions[agent, :]]
                    print(continuous_actions)
                    env.set_action_for_agent(behavior_name=behavior_name, agent_id=local_to_agent_map[agent], action=ActionTuple(discrete=None, continuous=np.array(continuous_actions)))
        env.step()
        # toto cele je nejaka kktina s tymi rewardmi
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Collect reward
        for agent in range(agent_count):
            local_agent = local_to_agent_map[agent]
            reward = 0
            if local_agent in terminal_steps:
                reward+=terminal_steps[local_agent].reward
            elif local_agent in decision_steps:
                reward+=decision_steps[local_agent].reward
            genomes[agent][1].fitness += reward
            total_reward += reward
            print(reward)

        if generation % 25 == 0: # save interval = 25
            print("\nSAVED PLOTS | GENERATION " + str(generation))
            visualize.plot_stats(stats, view=True, filename="NEAT/result/in_progress/recurrent-fitness"+str(generation)+".svg", label="CTRNN")
            visualize.plot_species(stats, view=True, filename="NEAT/result/in_progress/recurrent-speciation"+str(generation)+".svg", label="CTRNN")
        # save_progress(stats)
    global best_genome_current_generation
    best_genome_current_generation = max(genomes, key=lambda x: x[1].fitness)  # Save the best genome from this gen

    with open('NEAT/result/in_progress/best_genome'+str(generation)+'.pkl', 'wb') as f:
        pickle.dump(best_genome_current_generation, f)
    # Clean the environment for a new generation.
    env.reset()
    print("\nFinished generation")

if __name__ == "__main__":
    config_path = "NEAT/config_ctrnn"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    pop = neat.Population(config)
    # For saving checkpoints during training    Every 25th generation or 20 minutes
    pop.add_reporter(neat.Checkpointer(generation_interval=25, time_interval_seconds=1200, filename_prefix='checkpoints/NEAT-checkpoint-'))
    # Add reporter for fancy statistical result
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    evaluation = run_agent
    best_genome = pop.run(evaluation, max_generations)
    print(best_genome)
