from pathlib import Path

from logging_init import configure_logging
LOGGER = configure_logging(__file__)

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, base_dir='runs', run_name=None):
        if run_name is None:
            run_name = Path(__file__).resolve().stem

        self.run_name = run_name
        self.log_dir = Path(base_dir) / run_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        LOGGER.info(
            f"Initializing constructor for {self.__class__.__name__}, "
            f"Run name: {self.run_name}, "
            f"Log Dir: {self.log_dir}, "
            f"Writer: {self.writer}"
        )
        LOGGER.info("Run this command from the root directory to view metrics: tensorboard --logdir runs")


    def log_episode(self, agent_name, goal_index, episode, reward, steps, success, epsilon, loss=None, avg_reward_50=None, avg_steps_50=None, avg_success_50=None):
        tag_prefix = f"{agent_name}/goal_{goal_index}"
        self.writer.add_scalar(f"{tag_prefix}/episode_reward", reward, episode)
        self.writer.add_scalar(f"{tag_prefix}/episode_steps", steps, episode)
        self.writer.add_scalar(f"{tag_prefix}/success", success, episode)
        self.writer.add_scalar(f"{tag_prefix}/epsilon", epsilon, episode)

        if loss is not None:
            self.writer.add_scalar(f"{tag_prefix}/loss", loss, episode)

        if avg_reward_50 is not None:
            self.writer.add_scalar(f"{tag_prefix}/avg_reward_50", avg_reward_50, episode)

        if avg_steps_50 is not None:
            self.writer.add_scalar(f"{tag_prefix}/avg_steps_50", avg_steps_50, episode)

        if avg_success_50 is not None:
            self.writer.add_scalar(f"{tag_prefix}/avg_success_50", avg_success_50, episode)


    def log_test_result(self, agent_name, goal_index, success, steps, path):
        tag_prefix = f"{agent_name}/goal_{goal_index}"
        self.writer.add_scalar(f"{tag_prefix}/test_success", int(success), 0)
        self.writer.add_scalar(f"{tag_prefix}/test_steps", steps, 0)
        self.writer.add_text(f"{tag_prefix}/test_path", str(path), 0)

    def log_summary_scalar(self, name, value, step=0):
        self.writer.add_scalar(f"summary/{name}", value, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()