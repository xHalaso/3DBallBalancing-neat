import pickle
import sys
import numpy as np
import time
import atexit
import neat
import visualize
import math

# MLAGENTS stuff
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

save_nn_destination = 'NEAT/result/best.pkl'
result_destination= 'NEAT/result/in_progres/best_genome5.pkl'

# [PARAMETERS]  
max_generations = 1000 # Max number of generations
is_training = True   
no_graphics = False
is_multi = True
is_debug = False
load_from_checkpoint = False

generation = 0

# Best
best_current_gen = None

single_agent_env_path = "./Builds/SingleAgent/3DBallBalancing.exe"
multi_agent_env_path = "./Builds/MultipleAgents/3DBallBalancing.exe"

if is_multi:
        env = UnityEnvironment(file_name=multi_agent_env_path, worker_id=6, seed=0, no_graphics = no_graphics)
else:
    env = UnityEnvironment(file_name=single_agent_env_path, worker_id=5, seed=0, no_graphics = no_graphics)

# Reset the enviroment to get it ready  
print("ENV Has been reset")
env.reset()

behavior_specs = env.behavior_specs
behavior_name = list(behavior_specs)[0]
behavior_names = env.behavior_specs.keys()



print(f"Name of the behavior : {behavior_name}")
print("Number of observations : ", behavior_specs[behavior_name].observation_specs)
print(behavior_specs[behavior_name].observation_specs[0].observation_type)

# Define global variables to store running statistics
running_means = np.zeros(8)
running_variances = np.ones(8)
running_count = 1

def update_running_statistics(observation):
    global running_means, running_variances, running_count
    running_count += 1
    delta = observation - running_means
    running_means += delta / running_count
    running_variances += delta * (observation - running_means)

def normalize_observation(observation):
    global running_means, running_variances, running_count
    # Avoid division by zero in the first step
    if running_count > 1:
        stds = np.sqrt(running_variances / (running_count - 1))
        normalized_obs = (observation - running_means) / stds
    else:
        normalized_obs = observation - running_means
    return normalized_obs

def exit_handler():
    # visualize.plot_stats(stats, view=True, filename="NEAT/result/in_progress/recurrent-fitness"+str(generation)+".svg", label="CTRNN")
    # visualize.plot_species(stats, view=True, filename="NEAT/result/in_progress/recurrent-speciation"+str(generation)+".svg", label="CTRNN")
    with open(save_nn_destination, 'wb') as w:
        pickle.dump(best_current_gen, w)
    print("EXITING")
    env.close

atexit.register(exit_handler)

def run_agent(genomes, cfg):
    """
    Population size is configured as 12 to suit the training environment!
    :param genomes: All the genomes in the current generation.
    :param cfg: Configuration file
    :return: Best genome from generation.
    """
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    print(decision_steps, terminal_steps)
    # Keep track of agent ids for Unity and NEAT
    agent_to_local_map = {}
    local_to_agent_map = {}
    id_count = 0

    for step in decision_steps:
        # print(step)
        agent_to_local_map[step] = id_count
        local_to_agent_map[id_count] = step
        id_count += 1

    # Empty array to save all the neural networks
    policies = []

    for _, g in genomes:
        policy = neat.nn.FeedForwardNetwork.create(g, cfg)
        policies.append(policy)
        g.fitness = 0

    print("Population size (GENOMES): " + str(len(genomes)))

    global generation
    generation += 1
    done = False  # For the tracked_agent
    total_reward = 0.0

    # Agents
    agent_count = len(decision_steps.agent_id)
    print("Agent count: ", agent_count)

    terminal_agents = [] # these are agents that finished and removed

    if is_debug:
        input("Press Enter to star training...")

    while not done:
        actions = np.zeros(shape=(agent_count, 2))
        nn_input = np.zeros(shape=(agent_count, 8)) 
        
        # Decision step - agent requests action
        # Collect observations from the agents requesting input
        for agent in range(agent_count):  
            if local_to_agent_map[agent] in decision_steps:
                decision_steps = decision_steps
            else:
                continue
            step = decision_steps[local_to_agent_map[agent]]
            observation = np.concatenate(step.obs[:])
            update_running_statistics(observation)
            nn_input[agent] = normalize_observation(observation)
            # print(f"Input for Agent{agent}: ", nn_input[agent])

        # normalize inputs  
        # nn_input = 2 * ((nn_input - np.min(nn_input, axis=0)) / (np.max(nn_input, axis=0) - np.min(nn_input, axis=0))) - 1
        start = time.time()
        # Fetches actions by feed forward pass through the NNs
        # if len(decision_steps) > 0:  # More steps to take?
        for agent in range(agent_count):  # Iterates through all the agent indexes
            # Check if agent requests action
            if (local_to_agent_map[agent] in decision_steps):
                actions[agent] = policies[agent].activate(nn_input[agent])  # FPass
                # scaled_action = [a * 1 for a in action]
                # actions[agent] = scaled_action
                # print(f"Action for Agent{agent}: ", scaled_action)
                    
        end = time.time()
        time_spent_activating = (end - start)

        if len(decision_steps.agent_id) != 0:
            for agent in range(agent_count):
                # Check if agent requests action:
                if local_to_agent_map[agent] in decision_steps:
                    continuous_actions = [actions[agent, :]]
                    action_tuple = ActionTuple(discrete=None, continuous=np.array(continuous_actions))
                    if local_to_agent_map[agent] in decision_steps:
                        env.set_action_for_agent(behavior_name=behavior_name, agent_id=local_to_agent_map[agent], action=action_tuple)
        
        # Move the simulation forward
        env.step() # Does not mean 1 step in Unity. Runs until next decision step

        # toto cele je nejaka kktina s tymi rewardmi
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Remove agents that are terminal
        if terminal_steps:
            for step in terminal_steps: # the step is agent's id
                if step not in terminal_agents:
                    terminal_agents.append(step)

        # Collect reward
        for agent in range(agent_count):
            # Keep track of unity and neat id's
            local_agent = local_to_agent_map[agent]
            reward = 0

            if local_agent in terminal_steps:
                reward+=terminal_steps[local_agent].reward
            elif local_agent in decision_steps:
                reward+=decision_steps[local_agent].reward

            genomes[agent][1].fitness += reward
            total_reward += reward
            if reward > 10:
                print(
                    "Agent: " + str(agent) + " Fitness: " + str(genomes[agent][1].fitness) + " Reward: " + str(
                        reward))
            # print("Fitness: " , str(genomes[agent][1].fitness))

        if len(terminal_agents) >= agent_count:
            print([ta for ta in terminal_agents])
            print("--- [All agents are terminal!] ---")
            done = True
       # save_progress(stats)
    if len(decision_steps) != 0:
            # Reward status
            sys.stdout.write(
                "\rCollective reward: %.2f | Agents left: %d| Activation Time: %.2f" % (
                    total_reward,
                    len(decision_steps),
                    time_spent_activating))
            sys.stdout.flush()

    global best_genome_current_generation
    best_genome_current_generation = max(genomes, key=lambda x: x[1].fitness)  # Save the best genome from this gen
    
    if generation % 25 == 0: # save interval = 25
        print("\nSAVED PLOTS | GENERATION " + str(generation))
        # visualize.plot_stats(stats, view=True, filename="NEAT/result/in_progress/recurrent-fitness"+str(generation)+".svg", label="CTRNN")
        # visualize.plot_species(stats, view=True, filename="NEAT/result/in_progress/recurrent-speciation"+str(generation)+".svg", label="CTRNN")
        with open('NEAT/result/in_progress/best_genome'+str(generation)+'.pkl', 'wb') as f:
            pickle.dump(best_genome_current_generation, f)
    # Clean the environment for a new generation.
    env.reset()
    print("\nFinished generation")

# Run trained simulation
def run_agent_sim(genome, cfg):
    for gen in range(max_generations):
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        policy = neat.nn.RecurrentNetwork.create(genome, cfg)
        global generation
        generation += 1
        done = False  # For the tracked_agent
        agent_count = len(decision_steps.agent_id)
        agent_id = list(decision_steps)[0]
        while not done:
            actions = np.zeros(shape=(agent_count, 2))
            nn_input = np.concatenate((decision_steps[agent_id].obs[:])) 
            print(f"Input for Agent{agent_id}: ", nn_input)
            if len(decision_steps) > 0:  # More steps to take?
                action = policy.activate(nn_input)  # FPass for purple action
                print(f"Action for Agent{agent_id}: ", action)
            if len(decision_steps) > 0:
                # Check if agent requests action:
                continuous_actions = [action[:]]
                env.set_action_for_agent(behavior_name=behavior_name, agent_id=agent_id, action=ActionTuple(discrete=None, continuous=np.array(continuous_actions)))
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) > 0:
                agent_id = list(decision_steps)[0]

            # When whole teams are eliminated, end the generation.
            if len(decision_steps) == 0:
                done = True

        # Clean the environment for a new generation.
        env.reset()

if __name__ == "__main__":
    config_path = "NEAT/config_ctrnn"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    if is_training:
        pop = neat.Population(config)
        # For saving checkpoints during training    Every 25th generation or 20 minutes
        pop.add_reporter(neat.Checkpointer(generation_interval=25, time_interval_seconds=1200, filename_prefix='NEAT/checkpoints/NEAT-checkpoint-'))
        # Add reporter for fancy statistical result
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        evaluation = run_agent
        best_genome = pop.run(evaluation, max_generations)
        # Save best genome.
        with open(save_nn_destination, 'wb') as f:
            pickle.dump(best_genome, f)

        node_names = {
            -1: 'rotationX',
            -2: 'rotationZ',
            -3: 'positionDiffX',  # x component of the difference in position
            -4: 'positionDiffY',  # y component of the difference in position
            -5: 'positionDiffZ',  # z component of the difference in position
            -6: 'ballVelocityX',  # x component of the ball's velocity
            -7: 'ballVelocityY',  # y component of the ball's velocity
            -8: 'ballVelocityZ',  # z component of the ball's velocity
            0: 'outputRotationX',  # Output for rotation in the x axis
            1: 'outputRotationZ'   # Output for rotation in the z axis
        }
        visualize.draw_net(config, best_genome, True, node_names=node_names)

        visualize.draw_net(config, best_genome, view=True, node_names=node_names,
                            filename="NEAT/result/best_genome.gv")
        visualize.draw_net(config, best_genome, view=True, node_names=node_names,
                            filename="NEAT/result/best_genome-enabled.gv", show_disabled=False)
        visualize.draw_net(config, best_genome, view=True, node_names=node_names,
                            filename="NEAT/result/best_genome-enabled-pruned.gv", show_disabled=False, prune_unused=True)
    else:
        with open(save_nn_destination, "rb") as f:
            genome = pickle.load(f)
            print(genome)
        print(genome.fitness)
        run_agent_sim(genome, config)
