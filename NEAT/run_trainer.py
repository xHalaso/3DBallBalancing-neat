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
# save_nn_destination = 'NEAT/result/in_progress/best_genome125.pkl'
result_destination= 'NEAT/result/in_progres/best_genome5.pkl'
save_training_progress_prefix = 'NEAT/result/fitness/'

# [PARAMETERS]  
max_generations = 200 # Max number of generations
is_training = False   
no_graphics = is_training
is_multi = True
is_debug = False
load_from_checkpoint = False

generation = 0

# Best
best_current_gen = None

single_agent_env_path = "./Builds/SingleAgent/3DBallBalancing.exe"
multi_agent_env_path = "./Builds/60Agents/3DBallBalancing.exe"
engine_config_channel = EngineConfigurationChannel()

if is_training:#width=160, height=90, 
    engine_config_channel.set_configuration_parameters(width=2048 , height=1080, quality_level=0, time_scale=100)
else:
    engine_config_channel.set_configuration_parameters(width=2048 , height=1080)

if is_multi and is_training:
    env = UnityEnvironment(file_name=multi_agent_env_path, seed=0, no_graphics = no_graphics, side_channels=[engine_config_channel])
elif not is_training:   
    env = UnityEnvironment(file_name=single_agent_env_path, seed=0, no_graphics = no_graphics)

# Reset the enviroment to get it ready  
print("ENV Has been reset")
env.reset()

behavior_specs = env.behavior_specs
behavior_name = list(behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]

# Examine the number of observations per Agent
print("Number of observations : ", len(spec.observation_specs))

# Is there a visual observation ?
# Visual observation have 3 dimensions: Height, Width and number of channels
vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
print("Is there a visual observation ?", vis_obs)

# Is the Action continuous or multi-discrete ?
if spec.action_spec.continuous_size > 0:
  print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
  print(f"There are {spec.action_spec.discrete_size} discrete actions")

# For discrete actions only : How many different options does each action has ?
if spec.action_spec.discrete_size > 0:
  for action, branch_size in enumerate(spec.action_spec.discrete_branches):
    print(f"Action number {action} has {branch_size} different options")

def exit_handler():
    visualize.plot_stats(stats, view=True, filename="NEAT/result/in_progress/recurrent-fitness"+str(generation)+".svg", label="CTRNN")
    visualize.plot_species(stats, view=True, filename="NEAT/result/in_progress/recurrent-speciation"+str(generation)+".svg", label="CTRNN")
    with open("NEAT/result/on_exit/best_genome.pkl", 'wb') as w:
        pickle.dump(best_genome_current_generation, w)
    print("EXITING")
    env.close()

# Save training progress to files
def save_progress(statistics):
    statistics.save_genome_fitness(filename=save_training_progress_prefix+"genome_fitness.csv")
    statistics.save_species_count(filename=save_training_progress_prefix+"species_count.csv")
    statistics.save_species_fitness(filename=save_training_progress_prefix+"species_fitness.csv")

def run_agent(genomes, cfg):
    """
    Population size is configured as 12 to suit the training environment!
    :param genomes: All the genomes in the current generation.
    :param cfg: Configuration file
    :return: Best genome from generation.
    """
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    agent_count = len(decision_steps.agent_id)

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
        g.fitness = 0
        policy = neat.nn.RecurrentNetwork.create(g, cfg)
        policies.append(policy)

    print(f"Policies length {len(policies)}")

    #input("Press Enter to star training...")

    print("Population size (GENOMES): " + str(len(genomes)))

    global generation
    generation += 1
    done = False  # For the tracked_agent
    total_reward = 0.0

    # Agents
    print("Agent count: ", agent_count)

    terminal_agents = [] # these are agents that finished and removed

    if is_debug:
        input("Press Enter to star training...")

    done = [False if i < len(policies) else True for i in range(agent_count)]

    while not all(done):
        actions = np.ones(shape=(agent_count, 2))
        nn_input = np.zeros(shape=(agent_count, 8)) 
        
        # Decision step - agent requests action
        # Collect observations from the agents requesting input
        for agent in range(agent_count):  
            if local_to_agent_map[agent] in decision_steps:
                nn_input[agent] = np.asarray(decision_steps[local_to_agent_map[agent]].obs[:])
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
                    continuous_actions = np.array([actions[agent, :]])
                    # continuous_actions *= 2 + 0.5
                    continuous_actions *= 2
                    action_tuple = ActionTuple(discrete=None, continuous=continuous_actions)
                    env.set_action_for_agent(behavior_name=behavior_name, agent_id=local_to_agent_map[agent], action=action_tuple)
        
        # Move the simulation forward
        env.step() # Does not mean 1 step in Unity. Runs until next decision step

        # toto cele je nejaka kktina s tymi rewardmi
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Remove agents that are terminal
        # if terminal_steps:
        #     for step in terminal_steps: # the step is agent's id
        #         if step not in terminal_agents:
        #             terminal_agents.append(step)
        for agent in terminal_steps:
            done[local_to_agent_map[agent]] = True
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


    print("--- [All agents are terminal!] ---")

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
        save_progress(stats)
        with open('NEAT/result/in_progress/best_genome'+str(generation)+'.pkl', 'wb') as f:
            pickle.dump(best_genome_current_generation, f)
    # Clean the environment for a new generation.
    env.reset() #weedo need to do this as this is done in  unity itself
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
                continuous_actions = np.array([action[:]])
                continuous_actions *= 2
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
    atexit.register(exit_handler)
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
        visualize.plot_stats(stats, view=True, filename="NEAT/result/in_progress/recurrent-fitness"+str(generation)+".svg", label="CTRNN")
        visualize.plot_species(stats, view=True, filename="NEAT/result/in_progress/recurrent-speciation"+str(generation)+".svg", label="CTRNN")
        env.close()
    else:
        with open(save_nn_destination, "rb") as f:
            genome = pickle.load(f)
            print(genome)
        print(genome.fitness)
        run_agent_sim(genome, config)
