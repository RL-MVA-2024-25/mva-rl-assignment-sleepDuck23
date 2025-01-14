from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env import FastHIVPatient
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from copy import deepcopy
import time
from evaluate import evaluate_HIV, evaluate_HIV_population
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
import torch
import torch.nn as nn
    
class Deterministic_DQN(nn.Module):
    def __init__(self, input=6, output=4, layer = 256):
        super(Deterministic_DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, output)
            )
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class Stochastic_DQN(nn.Module):
    def __init__(self, input=6, output=4, layer = 256):
        super(Stochastic_DQN, self).__init__()

        self.linear_input = nn.Linear(input, layer)
        self.linear_normal = nn.Linear(layer, layer)
        self.batch_norm = nn.BatchNorm1d(layer)
        self.fc = nn.Sequential(
            nn.Linear(input, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            )
        self.mu_layer = nn.Linear(layer, output)
        self.std_layer = nn.Linear(layer, output)
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        std = F.softplus(log_std) 
        return mu, std
    
    def sample_action(self, x):
        mu, std = self(x) 
        action = mu + torch.randn_like(mu) * std
        return action

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) 
        self.data = []
        self.index = 0
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)  
env = TimeLimit(env=FastHIVPatient(domain_randomization=True), max_episode_steps=200)  

class ProjectAgent:
    config = {
          'learning_rate': 0.0008,
          'gamma': 0.99,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1,
          'epsilon_decay_period': 41000,
          'epsilon_delay_decay': 5000,
          'batch_size': 2000,
          'max_episode': 400,
          'nb_sample' : 15,
          'max_gradient_steps' : 8,
          'epsilon_seuil' : 0.2,
          'deterministic' : True,
          'episode_seuil' : 40,
          'explore_episodes' : 150,
          'patience_lr' : 7,
          'udpate_target_freq' : 500}
    dqn_network_deterministic = Deterministic_DQN()
    dqn_network_stochastic = Stochastic_DQN()
    def __init__(self):        

        self.deterministic = self.config['deterministic']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.deterministic == True:
            self.model_policy = self.dqn_network_deterministic.to(self.device)
            self.model_target = deepcopy(self.model_policy).to(self.device)
        else:
            self.model_policy = self.dqn_network_stochastic.to(self.device)
            self.model_target = deepcopy(self.model_policy).to(self.device)
        self.max_episode = self.config['max_episode']
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.memory = ReplayBuffer(self.config['buffer_size'], self.device)
        self.lr = self.config['learning_rate']
        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_stop = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.update_target_frequency = self.config['udpate_target_freq']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.max_gradient_steps = self.config['max_gradient_steps']
        self.patience = self.config['patience_lr']
        self.explore_episodes = self.config['explore_episodes']
        if self.deterministic == True:
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model_policy.parameters(), lr= self.lr)
        self.epsilon_seuil = self.config["epsilon_seuil"]
        self.scheduler  = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=self.patience, verbose= True)
        self.sample = self.config['nb_sample']
        self.episode_seuil = self.config['episode_seuil']
        
        self.gradient_time  = 0
        self.batch_time = 0
        self.map_time = 0
        self.compteur_stop = 0
        self.sampling_time = 0
        self.episode_time = 0
        self.epsilon  = self.epsilon_max
        self.step = 1
        self.gradient_steps = 0
        self.var = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.mu = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.previous_best = 0
        self.episode_seuil += self.explore_episodes

    def gaussian_nll_loss(self, mu, std, target):
        var = std ** 2  
        nll = 0.5 * torch.log(2 * torch.pi * var) + ((target - mu) ** 2) / (2 * var)
        return torch.mean(nll)

    def greedy_action(self, observation):
        observation = torch.Tensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            Q,  = self.model_policy(observation)
            return torch.argmax(Q).item()
        
    def Bayesian_TS(self, observation):
        observation = torch.Tensor(observation).unsqueeze(0).to(self.device)
        mu, std = self.model_policy(observation) 
        Q_sample = mu + torch.randn_like(mu) * std
        self.var = torch.mean(std)
        self.mu = torch.mean(mu)
        return torch.argmax(Q_sample).item()
    
    def act(self, observation, use_random=False):
        observation = np.sign(observation)*np.log(1+np.abs(observation))
        if self.deterministic == True:
            if use_random == True:
                if self.step > self.epsilon_delay:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)
                if np.random.rand() < self.epsilon:
                    action = env.action_space.sample()
                    return action
                else:
                    action = self.greedy_action(observation)   
            else:
                action = self.greedy_action(observation)           
            return action
        else:
            if use_random == False :
                action = self.Bayesian_TS(observation)
                
                return action
            if use_random==True:
                action = env.action_space.sample()
                return action
        
    def gradient_step(self, double_dqn): 
        start_sampling = time.perf_counter()
        X, A, R, Y, D  = self.memory.sample(self.batch_size) 
        X, A, R, Y, D = X.to(self.device, non_blocking=True), A.to(self.device, non_blocking=True), R.to(self.device, non_blocking=True), Y.to(self.device, non_blocking=True), D.to(self.device, non_blocking=True)
        R = torch.sign(R) * torch.log(1 + torch.abs(R))
        X = torch.sign(X) * torch.log(1 + torch.abs(X))
        Y = torch.sign(Y) * torch.log(1 + torch.abs(Y))
        self.sampling_time += time.perf_counter() - start_sampling
        
        if self.deterministic == True:
            if double_dqn :
                next_actions = self.model_policy(Y).argmax(dim=1)  
                QY_next = self.model_target(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
                update = R + self.gamma * QY_next * (1 - D)
                QXA = self.model_policy(X).gather(1, A.to(torch.long).squeeze(1)).unsqueeze(1)
                loss = self.criterion(QXA, update.unsqueeze(1).squeeze(1))
            else:
                QYmax = self.model_target(Y).max(1)[0].detach()
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                QXA = self.model_policy(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))
        else:
            if double_dqn :
                next_actions = self.model_policy.sample_action(Y).argmax(1, keepdim=True)
                QYmax = self.model_target.sample_action(Y).gather(1, next_actions).squeeze(1).detach()
                update = R + (1 - D) * self.gamma * QYmax
                QXA = self.model_policy.sample_action(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))
            else:
                QYmax = self.model_target.sample_action(Y).max(1)[0].detach()
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)

                QXA = self.model_policy.sample_action(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        

    
    def early_stop(self, best_score):

        if self.compteur_stop == int(self.max_gradient_steps*self.patience) +2:
            return True
        if abs(best_score - self.previous_best) < 1.1:
            self.compteur_stop += 1
        else:
            self.compteur_stop = 0
            self.previous_best = best_score
        return False
        
    def gradient_steps_calculation(self, episode):
        k = 2  
        min_steps = 1  
        max_steps = self.max_gradient_steps 
        if self.deterministic == True:
            if self.epsilon >0.55:
                return 1

            elif self.epsilon > self.epsilon_seuil:
                scale = np.exp(-k * (self.epsilon - self.epsilon_seuil) / (1 - self.epsilon_seuil))
                self.gradient_steps =  int(min_steps + (max_steps - min_steps) * scale)
            else:
                if self.compteur_stop % self.patience == 0  and self.compteur_stop !=0 and self.gradient_steps !=1:
                    self.gradient_steps = self.gradient_steps -1

        else:
            if episode < self.explore_episodes:
                self.gradient_steps = 0
            if self.explore_episodes < episode <= self.episode_seuil:
                scale = 1-np.exp(-k * (episode - self.explore_episodes) / (self.episode_seuil - self.explore_episodes))
                self.gradient_steps = int(min_steps + (max_steps) * scale)
            else:
                if self.compteur_stop % self.patience == 0  and self.compteur_stop !=0 and self.gradient_steps !=1:
                    self.gradient_steps = self.gradient_steps -1

    def train(self, env):
        episode = 0
        episode_return, var_return = [], []
        state, _ = env.reset()
        episode_cum_reward = 0
        best_score = 0        
        cumulated_var = 0
        cumulated_mu = 0

        while episode < self.max_episode:

            if episode != 0:
                if trunc == True :
                    self.episode_time = time.perf_counter()
                    torch.cuda.synchronize()
            if self.deterministic == False:
                if episode <= self.explore_episodes:
                    use_random = True
                else:
                    use_random = False
            else:
                use_random = True

            if self.deterministic == True:
                action = self.act(state, use_random=use_random)
            else:
                action = self.act(state, use_random=use_random)
            
            next_state, reward, done, trunc, _ = env.step(action)

            self.memory.append(state , action, reward,next_state, trunc) 
            
            episode_cum_reward += reward
            
            cumulated_var += self.var
            cumulated_mu += self.mu

            if trunc == True:
                self.gradient_steps_calculation(episode)
            

            if self.deterministic == True:
                if self.epsilon < self.epsilon_seuil and trunc == True:
                    stop = self.early_stop(best_score)
                    self.scheduler.step(best_score)
                    if stop :
                        print(f"Best score {best_score:.2e}")
                        return episode_return
            else:
                if episode > self.episode_seuil and trunc == True:
                    stop = self.early_stop(best_score)
                    self.scheduler.step(best_score)
                    if stop :
                        print(f"Best score {best_score:.2e}")
                        return episode_return
                    
            if len(self.memory) > self.batch_size:    
                for i in range(self.gradient_steps):
                    self.gradient_step(double_dqn=False)
            

            if self.step % self.update_target_frequency  == 0:
                self.model_target.load_state_dict(self.model_policy.state_dict())

            if done or trunc :           
                episode += 1
                episode_return.append(episode_cum_reward)
                var_return.append(cumulated_var)
                if best_score > 8000000000 :  

                    self.model_policy.eval()
                    validation_score_hiv = evaluate_HIV(agent=self, nb_episode=5)
                    validation_score_population = evaluate_HIV_population(agent=self, nb_episode=20)
                    self.model_policy.train()

                    if validation_score_hiv + validation_score_population > best_score:
                        best_score = validation_score_hiv + validation_score_population
                        self.save("policy")
                else:
                    validation_score_hiv = 0
                    validation_score_population = 0
                    if episode_cum_reward > best_score:
                        best_score = episode_cum_reward
                        self.save("policy")

                if self.deterministic:
                    epsilon_or_variance = f"Epsilon {self.epsilon:6.4f} | "
                else:
                    epsilon_or_variance = f"Variance {cumulated_var.cpu().detach().numpy().item():.2e} | Mean {cumulated_mu.cpu().detach().numpy().item():.2e} | "
                torch.cuda.synchronize()
                print(f"Episode {episode} | ",
                    epsilon_or_variance,
                    f"Ep return {episode_cum_reward:.2e} | ",   
                    f"HIV {validation_score_hiv:.2e} | ",
                    f"Population {validation_score_population:.2e} | ",                 
                    f"Episode time {(time.perf_counter() - self.episode_time):1.1f} | ",
                    f"Gradient steps {self.gradient_steps}",
                    
                    sep='')
                self.gradient_time = 0
                self.sampling_time = 0
                self.map_time = 0
                self.batch_time = 0
                env_duration = 0
                state, _ = env.reset()



                episode_cum_reward = 0
                cumulated_var = 0
                cumulated_mu = 0

            else:
                state = next_state
            self.step += 1
        print(f"Best score {best_score:.2e}")
        return episode_return


    def save(self, path):
        os.makedirs(path, exist_ok=True)  
        torch.save(self.model_policy.state_dict(), os.path.join(path, "model.pth"))
        print(f"Model saved to {os.path.join(path, 'model.pth')}")

    def load(self):
        file_path = os.path.join(os.getcwd(), 'policy', 'model.pth')
        print(file_path)

        self.model_policy.load_state_dict(torch.load(file_path, map_location=torch.device('cuda'))) #cuda
        self.model_policy.eval()
        print(f"Model loaded from {file_path}")
        return 







