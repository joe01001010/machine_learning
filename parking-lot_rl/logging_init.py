import logging
from pathlib import Path


def configure_logging(source_file, log_dir='logs', level=logging.INFO):
    """
    Configure a shared file logger for the current process and return a
    module-specific logger.
    """
    source_path = Path(source_file).resolve()
    module_name = source_path.stem
    logs_path = Path(log_dir).resolve()
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / f"{module_name}.log"

    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def summarize_agent(agent_name, goal_index, goal, history, train_seconds, test_success, test_steps, final_window=50):
    rewards = history["episode_rewards"]
    steps = history["episode_steps"]
    successes = history["success"]
    losses = [loss for loss in history["loss"] if loss is not None]

    final_rewards = rewards[-final_window:] if len(rewards) >= final_window else rewards
    final_steps = steps[-final_window:] if len(steps) >= final_window else steps
    final_successes = successes[-final_window:] if len(successes) >= final_window else successes

    summary = {
        "agent": agent_name,
        "goal_index": goal_index,
        "goal": goal,
        "episodes_run": len(successes),
        "training_successes": sum(successes),
        "overall_success_rate": sum(successes) / len(successes) if successes else 0.0,
        "final_avg_reward": sum(final_rewards) / len(final_rewards) if final_rewards else 0.0,
        "final_avg_steps": sum(final_steps) / len(final_steps) if final_steps else 0.0,
        "final_success_rate": sum(final_successes) / len(final_successes) if final_successes else 0.0,
        "final_epsilon": history["epsilon"][-1] if history["epsilon"] else None,
        "final_avg_loss": sum(losses[-final_window:]) / len(losses[-final_window:]) if losses else None,
        "training_time_seconds": train_seconds,
        "test_success": int(test_success),
        "test_steps": test_steps,
    }
    return summary


def summarize_by_agent(pipeline_metrics):
    grouped = {}

    for row in pipeline_metrics:
        agent = row["agent"]
        if agent not in grouped:
            grouped[agent] = []
        grouped[agent].append(row)

    summary_rows = []
    for agent, rows in grouped.items():
        summary_rows.append({
            "agent": agent,
            "avg_episodes_run": sum(r["episodes_run"] for r in rows) / len(rows),
            "avg_overall_success_rate": sum(r["overall_success_rate"] for r in rows) / len(rows),
            "avg_final_reward": sum(r["final_avg_reward"] for r in rows) / len(rows),
            "avg_final_steps": sum(r["final_avg_steps"] for r in rows) / len(rows),
            "avg_final_success_rate": sum(r["final_success_rate"] for r in rows) / len(rows),
            "avg_training_time_seconds": sum(r["training_time_seconds"] for r in rows) / len(rows),
            "avg_test_steps": sum(r["test_steps"] for r in rows) / len(rows),
            "avg_final_loss": (
                sum(r["final_avg_loss"] for r in rows if r["final_avg_loss"] is not None) /
                len([r for r in rows if r["final_avg_loss"] is not None])
            ) if any(r["final_avg_loss"] is not None for r in rows) else None,
        })
    return summary_rows