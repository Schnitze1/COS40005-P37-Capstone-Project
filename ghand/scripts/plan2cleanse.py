import numpy as np
import copy
import gym

class Plan2CleanseWrapper(gym.Wrapper):
    """
    Plan2Cleanse: Test-Time Backdoor Defense Wrapper.
    Uses Monte Carlo rollouts (via physics state save/restore) to detect trigger states 
    and plan actions that maximize clean returns and evade backdoor activation.
    """
    def __init__(self, env, agent, algo_name, horizon=3, num_rollouts=15, noise_std=0.1, 
                 trigger_indices=None, trigger_values=None, trigger_threshold=0.1):
        super(Plan2CleanseWrapper, self).__init__(env)
        self.agent = agent
        self.algo_name = algo_name
        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.noise_std = noise_std
        
        # Trigger definition for oracle mitigation mode
        self.trigger_indices = trigger_indices
        self.trigger_values = trigger_values
        self.trigger_threshold = trigger_threshold
        
        # Statistics
        self.total_overrides = 0
        self.total_steps = 0

    def get_env_state(self):
        """Extract the complete physical simulator state."""
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "sim"):
            # MuJoCo style
            return unwrapped.sim.get_state()
        elif hasattr(unwrapped, "get_state"):
            # SAPIEN/ManiSkill2 style
            return unwrapped.get_state()
        else:
            return copy.deepcopy(unwrapped)

    def set_env_state(self, state):
        """Restore the complete physical simulator state."""
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "sim"):
            # MuJoCo style
            unwrapped.sim.set_state(state)
        elif hasattr(unwrapped, "set_state"):
            # SAPIEN/ManiSkill2 style
            unwrapped.set_state(state)

    def is_trigger_state(self, obs):
        """Check if observation features match the backdoor trigger signature."""
        if self.trigger_indices is not None and self.trigger_values is not None:
            trigger_feat = obs[self.trigger_indices]
            dist = np.linalg.norm(trigger_feat - self.trigger_values)
            return dist < self.trigger_threshold
        return False

    def select_action(self, obs):
        """
        Uses MCTS-style lookahead to evaluate actions and override the policy 
        if it selects a poisoned action or enters a trigger zone.
        """
        self.total_steps += 1
        
        # Get base policy action
        # Shape handling: d3rlpy models expect batch dimension
        state_input = np.expand_dims(obs, axis=0)
        base_action = self.agent.predict(state_input)[0]
        
        # Check if the current state is already in the trigger zone
        is_triggered = self.is_trigger_state(obs)
        
        # If not triggered, return base action directly (skip planning)
        if not is_triggered:
            return base_action
            
        # Perform Monte Carlo Rollouts to evaluate action candidates
        best_action = base_action
        best_score = -float('inf')
        override_chosen = False
        
    def get_wrapper_states(self, env):
        """Recursively retrieve time/step counter attributes from all Gym wrappers."""
        states = {}
        curr_env = env
        while True:
            for attr in ["_elapsed_steps", "_elapsed_seconds", "steps", "_steps", "num_steps"]:
                if hasattr(curr_env, attr):
                    states[id(curr_env)] = (curr_env, attr, getattr(curr_env, attr))
            if hasattr(curr_env, "env"):
                curr_env = curr_env.env
            else:
                break
        return states

    def restore_wrapper_states(self, states):
        """Restore wrapper time/step counters to original values."""
        for curr_env, attr, val in states.values():
            setattr(curr_env, attr, val)

    def select_action(self, obs):
        """
        Uses MCTS-style lookahead to evaluate actions and override the policy 
        if it selects a poisoned action or enters a trigger zone.
        """
        self.total_steps += 1
        
        # Get base policy action
        # Shape handling: d3rlpy models expect batch dimension
        state_input = np.expand_dims(obs, axis=0)
        base_action = self.agent.predict(state_input)[0]
        
        # Check if the current state is already in the trigger zone
        is_triggered = self.is_trigger_state(obs)
        
        # If not triggered, return base action directly (skip planning)
        if not is_triggered:
            return base_action
            
        # Perform Monte Carlo Rollouts to evaluate action candidates
        best_action = base_action
        best_score = -float('inf')
        override_chosen = False
        
        # Save initial environment state (both physics and wrapper counters)
        init_physics = self.get_env_state()
        init_wrappers = self.get_wrapper_states(self.env)
        
        # Generate action candidates
        action_dim = self.env.action_space.shape[0]
        candidates = []
        
        # Candidate 0: The base policy action
        candidates.append(base_action)
        
        # Candidates 1..N-2: Perturbed policy actions
        for _ in range(self.num_rollouts - 2):
            noise = np.random.normal(0, self.noise_std, size=action_dim)
            perturbed_action = np.clip(base_action + noise, self.env.action_space.low, self.env.action_space.high)
            candidates.append(perturbed_action)
            
        # Candidate N-1: Fully random action
        candidates.append(self.env.action_space.sample())
        
        # Evaluate each candidate action sequence
        for idx, candidate_act in enumerate(candidates):
            score = 0.0
            triggered_in_rollout = False
            
            # Restore environment to initial state before each rollout
            self.set_env_state(init_physics)
            self.restore_wrapper_states(init_wrappers)
            
            # Step 1: Execute the candidate action
            next_obs, reward, done, info = self.env.step(candidate_act)
            score += reward
            if self.is_trigger_state(next_obs):
                triggered_in_rollout = True
                
            # Steps 2..H: Follow policy for subsequent steps
            curr_obs = next_obs
            for t in range(1, self.horizon):
                if done:
                    break
                # Predict next action using the agent
                act = self.agent.predict(np.expand_dims(curr_obs, axis=0))[0]
                curr_obs, reward, done, info = self.env.step(act)
                score += reward
                if self.is_trigger_state(curr_obs):
                    triggered_in_rollout = True
            
            # Penalize trigger activation heavily
            if triggered_in_rollout:
                score -= 1000.0
                
            # Track best action
            if score > best_score:
                best_score = score
                best_action = candidate_act
                if idx > 0:
                    override_chosen = True
                    
        # Restore environment back to the starting state before returning
        self.set_env_state(init_physics)
        self.restore_wrapper_states(init_wrappers)
        
        # Log overrides
        if override_chosen:
            self.total_overrides += 1
            
        return best_action

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs

    def step(self, action):
        # Intercept action and select safe action based on current observation
        safe_action = self.select_action(self.current_obs)
        obs, reward, done, info = self.env.step(safe_action)
        self.current_obs = obs
        return obs, reward, done, info

