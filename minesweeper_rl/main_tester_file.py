# main_tester_file.py
from agent_model import *
from minesweeper_env import *
from collections import deque
import random, torch, time
import numpy as np
import torch.nn as nn
import torch.optim as optim

def main():
    rows, cols = 9, 9
    mines = 6
    num_episodes = 30000
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999998
    TARGET_UPDATE_FREQ = 1000
    BATCH_SIZE = 64
    PRETRAINED_MODEL_PATH = "minesweeper_trained_model.pth"
    
    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=mines)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_pretrained_model(PRETRAINED_MODEL_PATH, rows, cols, device)
    
    # Initialize target model from the loaded model
    target_model = AgentModel(rows=rows, cols=cols).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    replay_buffer = deque(maxlen=20000)
    
    # More conservative learning rates for fine-tuning
    optimizer = optim.AdamW([
        {'params': model.conv_layers.parameters(), 'lr': 5e-5},
        {'params': model.attention.parameters(), 'lr': 5e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='max',
                                                    factor=0.5,
                                                    patience=200,
                                                    min_lr=1e-6)
    
    criterion = nn.SmoothL1Loss()

    recent_wins = 0
    total_rewards = []
    step_counter = 0

    for episode in range(num_episodes):
        render = (episode + 1) % num_episodes == 0
        obs, info = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        if episode + 1 == 1 or episode % 100 == 0:
            episode_wins = 0

        while not done:
            state = preprocess_obs(obs).to(device)
            q_values = model(state).squeeze(0)
            valid_actions = env._get_valid_actions()
            current_epsilon = max(epsilon_min, epsilon * (epsilon_decay ** episode))
            action = get_strategic_action(q_values, valid_actions, state, current_epsilon, rows, cols)
            action_idx = action[0] * env.cols + action[1]

            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_steps += 1

            # Store experience
            next_state = preprocess_obs(next_obs)
            replay_buffer.append((
                state.cpu(), 
                action_idx, 
                reward, 
                next_state.cpu(), 
                done,
            ))

            # Training phase with gradient accumulation
            if len(replay_buffer) >= BATCH_SIZE and episode_steps % 2 == 0:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                
                states = torch.cat([item[0] for item in batch]).to(device)
                actions = torch.tensor([item[1] for item in batch], dtype=torch.long).to(device)
                rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(device)
                next_states = torch.cat([item[3] for item in batch]).to(device)
                dones = torch.tensor([item[4] for item in batch], dtype=torch.bool).to(device)

                # Current Q values
                current_q_values = model(states)
                current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Next Q values from target network with double DQN
                with torch.no_grad():
                    next_actions = model(next_states).max(1)[1]
                    next_q_values = target_model(next_states)
                    next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards + gamma * next_q * (~dones).float()

                # Compute loss and optimize
                loss = criterion(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Tighter gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            # Update target network less frequently
            step_counter += 1
            if step_counter % TARGET_UPDATE_FREQ == 0:
                target_model.load_state_dict(model.state_dict())

            # Render if needed
            if render:
                print(f"\nEpisode {episode + 1}, Step {episode_steps}")
                print("Q-values:")
                q_vals_2d = q_values.detach().cpu().numpy().reshape(rows, cols)
                for r in range(rows):
                    for c in range(cols):
                        print(f"{q_vals_2d[r, c]:7.2f}", end=" ")
                    print()
                print(f"Action: ({action[0]+1}, {action[1]+1}), Reward: {reward:.2f}")
                print(f"Action index: {action_idx}")
                env.render()
                time.sleep(0.5)

            obs = next_obs
            if info.get('won', False):
                episode_wins += 1

        # End of episode
        epsilon = max(epsilon_min, epsilon * (epsilon_decay ** episode))
        total_rewards.append(total_reward)
        
        if info.get('won', False):
            recent_wins += 1

        # Logging and early stopping
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            win_rate = recent_wins / 100.0
            
            # Update learning rate based on performance
            scheduler.step(win_rate)
            
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Win Rate: {win_rate:.2f}, "
                  f"Epsilon: {current_epsilon:.3f}")

            recent_wins = 0

    final_avg_reward = np.mean(total_rewards[-100:])
    print(f"\nTraining completed")
    print(f"Average Reward for last 100 episodes: {final_avg_reward:.2f}")
    print(f"Win rate for last 100 games: {int(win_rate * 100)}/100")
    
    # Save the final model
    torch.save(model.state_dict(), "minesweeper_final_model.pth")
    print("Model saved as 'minesweeper_final_model.pth'")

def preprocess_obs(obs):
    """
    Improved observation preprocessing with better normalization
    """
    board = obs['board']
    visible = obs['visible']
    H, W = board.shape
    state = np.zeros((5, H, W), dtype=np.float32)

    # Channel 0: visible mask (1 if revealed, 0 if hidden)
    state[0] = visible.astype(np.float32)

    # Channel 1: revealed numbers with better encoding
    revealed_nums = board.astype(np.float32)
    
    # Normalize numbers 0-8 to [0, 1]
    number_mask = (board >= 0) & (board <= 8)
    revealed_nums[number_mask] = board[number_mask] / 8.0
    
    # Hidden cells remain 0
    revealed_nums[~visible] = 0.0
    state[1] = revealed_nums

    # Channel 2: Mine probability estimation
    mine_probs = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            if visible[r, c] and board[r, c] > 0:
                # For revealed numbers, calculate mine probability in neighbors
                hidden_neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < H and 0 <= nc < W and 
                            not visible[nr, nc] and (nr, nc) != (r, c)):
                            hidden_neighbors.append((nr, nc))
                
                if hidden_neighbors:
                    prob = board[r, c] / len(hidden_neighbors)
                    for nr, nc in hidden_neighbors:
                        mine_probs[nr, nc] = max(mine_probs[nr, nc], prob)
    state[2] = mine_probs

    # Channel 3: safe move probabilities (inverse of mine probs)
    state[3] = 1.0 - mine_probs

    # Channel 4: frontier cells (hidden cells adjacent to revealed numbers)
    frontier = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            if not visible[r, c]:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < H and 0 <= nc < W and 
                            visible[nr, nc] and board[nr, nc] > 0):
                            frontier[r, c] = 1.0
                            break
    state[4] = frontier
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


def load_pretrained_model(model_path, rows, cols, device):
    """Load a pretrained model for continued training"""
    model = AgentModel(rows=rows, cols=cols).to(device)
    
    try:
        # Load the saved model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded pretrained model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Starting with new model.")
    except Exception as e:
        print(f"Error loading model: {e}. Starting with new model.")
        model = AgentModel(rows=rows, cols=cols).to(device)
    
    return model


def get_strategic_action(q_values, valid_actions, state, epsilon, rows, cols):
        """Choose actions with preference for safe moves"""
        
        if np.random.rand() < epsilon:
            # During exploration, prefer frontier cells
            frontier_actions = []
            for action in valid_actions:
                r, c = action
                if state[0, 4, r, c] > 0.5:  # Frontier cell from channel 4
                    frontier_actions.append(action)
            
            if frontier_actions:
                return random.choice(frontier_actions)
            else:
                return random.choice(valid_actions)
        else:
            # During exploitation, use Q-values but prefer high safety probability
            q_values_np = q_values.detach().cpu().numpy()
            action_scores = []
            
            for (r, c) in valid_actions:
                idx = r * cols + c
                base_score = q_values_np[idx]
                
                # Boost score for high safety probability
                safety_boost = state[0, 3, r, c].item() * 2.0  # Channel 3: safe probabilities
                
                # Boost score for frontier cells
                frontier_boost = state[0, 4, r, c].item() * 1.0
                
                total_score = base_score + safety_boost + frontier_boost
                action_scores.append((total_score, idx, (r, c)))
            
            # Choose action with highest combined score
            best_score, best_idx, best_action = max(action_scores)
            return best_action


if __name__ == "__main__":
    main()