from omegaconf import OmegaConf

from src.data.agents import QLearningAgent, BeliefStateAgent
from src.data.environments import Block2AFCTask
from src.visualization.visualize import plot_psychometric_curve


class Experiment:
    def __init__(self, config):
        self.config = OmegaConf.load(config)

        if self.config["environment"].lower() == "block2afctask":
            self.blocks = self.config["blocks"]
            self.environment = Block2AFCTask(self.blocks)
        else:
            raise NotImplementedError("Block2AFCTask is the only implemented environment!")

        if self.config["agent"].lower() == "qlearningagent":
            self.learning_rate = self.config["learning_rate"]
            self.epsilon = self.config["epsilon"]
            if not (0 <= self.epsilon <= 1):
                raise ValueError("Epsilon should be between 0 and 1!")
            self.agent = QLearningAgent(self.learning_rate, self.epsilon)
        elif self.config["agent"].lower() == "beliefstateagent":
            self.p_reward = self.config["p_reward"]
            self.p_switch = self.config["p_switch"]
            if not (0 <= self.p_reward <= 1 and 0 <= self.p_switch <= 1):
                raise ValueError("p_reward and p_switch should be between 0 and 1!")
            self.agent = BeliefStateAgent(self.p_reward, self.p_switch)
        else:
            raise NotImplementedError("QLearningAgent and BeliefStateAgent are the only implemented agents!")

    def get_blocks(self):
        return self.blocks

    def run(self):
        """Run an agent through an experiment with parameters set by block_params."""
        for ep in range(self.environment.total_trials + 1):
            self.environment.step()
            if self.environment.done:
                break
            stimuli = self.environment.get_current_stimuli()
            agent_action = self.agent.sample_action(stimuli)
            correct_action = self.environment.get_current_reward()
            reward = (agent_action == correct_action)
            self.agent.update(stimuli, agent_action, reward)

    def plot_psychometric(self, save=False, path=None):
        if not self.environment.done:
            raise Exception("Run experiment before plotting results!")

        blocks = self.blocks
        actions = self.agent.action_history
        stimuli = self.environment.stimulus_history
        rewards = self.environment.reward_history

        fig = plot_psychometric_curve(blocks, rewards, stimuli, actions)
        if save and path:
            fig.savefig(path)
        elif save:
            raise ValueError("Save location not specified!")
