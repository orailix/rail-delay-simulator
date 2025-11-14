# Implementation inspired from https://github.com/toshikwa/gail-airl-ppo.pytorch

import os
import math
import torch
import gc
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from tqdm import tqdm
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from collections import deque
from functools import partial

from src.utils.logger import CustomCSVLogger, MetricsLoggingCallback
from src.utils.utils import load_pickle, save_pickle, get_subdir_path, get_dates, setup_run_folder
from src.environment.simulation import Simulator, load_itineraries_from_dates
from src.utils.metrics import simulate_and_compute_mae
from src.utils.weightssaver import WeightSaver
from src.models.transformer import Transformer

class GAIL(pl.LightningModule):
    """ 
    Generative Adversarial Imitation Learning (GAIL) implementation with a Transformer policy, discriminator and critic.

    Args:
        model_config (dict): Transformer config for the policy (expects 'input_dim', 'num_classes').
        dataset_config (dict): Dataset configuration parameters for data loading.
        sim_config (dict): Simulation configuration (trajectories length, simulation batch size, ...).
        lr_p (float): Policy learning rate.
        weight_decay_p (float): Policy weight decay.
        lr_d (float): Discriminator learning rate.
        weight_decay_d (float): Discriminator weight decay.
        lr_c (float): Critic learning rate.
        weight_decay_c (float): Critic weight decay.
        disc_steps (int): Discriminator steps per iteration.
        ppo_steps (int): Policy (PPO) steps per iteration.
        smoothing_eps (float): Label-smoothing ε for discriminator targets.
        ent_max (float): Initial entropy coefficient.
        ent_min (float): Minimum entropy coefficient.
        uses_pretrained_bc (bool): If True, initialize policy from BC.
        itineraries (dict): Dictionary of train itineraries used in simulation.
        sc (dict): Data scheme.
        cat (dict): Categorical Categorical feature definitions.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        eval_config (dict): Configuration for evaluation.
    """
    def __init__(self, model_config: dict, dataset_config: dict, sim_config: dict, lr_p: float, weight_decay_p: float, lr_d: float, weight_decay_d: float, lr_c: float, weight_decay_c: float, 
    disc_steps: int, ppo_steps: int, smoothing_eps: float, ent_max: float, ent_min: float, uses_pretrained_bc: bool, itineraries: dict, sc: dict, cat: dict, stations_emb: dict, lines_emb: dict, 
    eval_config: dict) -> None:
        super().__init__()

        self.policy = Transformer(**model_config)

        state_dim, action_dim = model_config['input_dim'], model_config['num_classes']
        
        disc_config = model_config.copy()
        disc_config['input_dim'] = state_dim + action_dim
        disc_config['num_classes'] = 1

        value_config = model_config.copy()
        value_config['input_dim'] = state_dim
        value_config['num_classes'] = 1

        self.discriminator = Transformer(**disc_config)
        self.critic = Transformer(**value_config)

        self.dataset_config = dataset_config
        self.sim_config = sim_config
        self.traj_len = sim_config['traj_len']
        self.disc_steps = disc_steps
        self.ppo_steps = ppo_steps

        self.lr_p = lr_p 
        self.weight_decay_p = weight_decay_p 
        self.lr_d = lr_d
        self.weight_decay_d = weight_decay_d
        self.lr_c = lr_c 
        self.weight_decay_c = weight_decay_c

        self.gamma = math.exp(math.log(0.05) / self.traj_len)
        self.lambd = self.gamma
        self.max_grad_norm = 0.5
        self.smoothing_eps = smoothing_eps
        self.ent_max = ent_max
        self.ent_min = ent_min
        self.coef_ent = ent_max
        self.clip_eps = 0.2
        self.uses_pretrained_bc = uses_pretrained_bc

        self.itineraries = itineraries
        self.horizon_obs_bins = eval_config['horizon_obs_bins']
        self.delay_delta_bins = eval_config['delay_delta_bins']
        self.nb_traj_eval = eval_config['nb_traj']
        self.pred_horizon = eval_config['pred_horizon']
        self.sc = sc
        self.cat = cat
        self.stations_emb = stations_emb
        self.lines_emb = lines_emb
        
        self.automatic_optimization = False

    def configure_optimizers(self) -> list:
        """
        Configure AdamW optimizer for the policy, discriminator and critic.

        Returns:
            list[torch.optim.Optimizer]: List with the 3 AdamW optimizers.
        """
        policy_optimizer = optim.AdamW(
            self.policy.parameters(), 
            lr=self.lr_p, 
            weight_decay=self.weight_decay_p
        )
        discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(), 
            lr=self.lr_d, 
            weight_decay=self.weight_decay_d
        )
        critic_optimizer = optim.AdamW(
            self.critic.parameters(), 
            lr=self.lr_c, 
            weight_decay=self.weight_decay_c
        )
        
        return [policy_optimizer, discriminator_optimizer, critic_optimizer]

    def update_parameters(self) -> None:
        """
        Update entropy coefficient based on current epoch.

        Returns:
            None
        """
        halfway_point = self.trainer.max_epochs / 2.0
        epoch = float(self.current_epoch)

        frac = min(epoch / halfway_point, 1.0)
        self.coef_ent = self.ent_max - frac * (self.ent_max - self.ent_min)
    
    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Not used because of GAIL complex training scheme.
        """
        pass

def on_validation_epoch_start(self) -> None:
        """
        Initialize validation metrics at epoch start.
        """
        self.mae_delay_val = []
        self.mae_horizon_val = []
        self.counter_delay_val = []
        self.counter_horizon_val = []
        self.sse_val = 0

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Perform one validation step by simulating trajectories and computing metrics.

        Args:
            batch (tuple): (initial_states, metadatas) for the trajectories.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        initial_states, metadatas = batch
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = simulate_and_compute_mae(initial_states, metadatas, self.policy, self.itineraries, True, self.nb_traj_eval, self.pred_horizon, 'median', self.dataset_config['nb_future_station_reg'], self.device, self.sc, self.cat, self.stations_emb, self.lines_emb, self.dataset_config, self.delay_delta_bins, self.horizon_obs_bins, 'transformer')

        self.mae_delay_val += mae_delay
        self.mae_horizon_val += mae_horizon
        self.counter_delay_val += counter_delay
        self.counter_horizon_val += counter_horizon
        self.sse_val += sse


    def on_validation_epoch_end(self) -> None:
        """
        Aggregate and log validation metrics at the end of an epoch.

        Computes weighted MAE per delay/horizon bin, overall MAE, and MSE/RMSE
        from accumulated values across validation steps.

        Returns:
            None
        """
        sum_del_counter = torch.stack(self.counter_delay_val).sum(dim=0)
        sum_hor_counter = torch.stack(self.counter_horizon_val).sum(dim=0)
        
        mae_delay_tensor = torch.stack(self.mae_delay_val)
        counter_delay_tensor = torch.stack(self.counter_delay_val)
        
        mae_horizon_tensor = torch.stack(self.mae_horizon_val)
        counter_horizon_tensor = torch.stack(self.counter_horizon_val)
        
        mae_delay = (mae_delay_tensor * counter_delay_tensor).sum(dim=0) / sum_del_counter.clamp(min=1)
        mae_horizon = (mae_horizon_tensor * counter_horizon_tensor).sum(dim=0) / sum_hor_counter.clamp(min=1)

        for i in range(len(mae_horizon)):
            self.log(f"hor{i}",mae_horizon[i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        for i in range(len(mae_delay)):
            self.log(f"del{i}",mae_delay[i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        mae = torch.dot(mae_horizon, sum_hor_counter) / sum_hor_counter.sum()
        
        self.log(f"mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        mse = self.sse_val/ sum_hor_counter.sum()

        self.log(f"mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log(f"rmse", np.sqrt(mse), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def on_test_epoch_start(self) -> None:
        """
        Initialize test metrics at epoch start.
        """
        self.mae_delay_test = []
        self.mae_horizon_test = []
        self.counter_delay_test = []
        self.counter_horizon_test = []
        self.sse_test = 0

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Perform one test step by simulating trajectories and computing metrics.

        Args:
            batch (tuple): (initial_states, metadatas) for the trajectories.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        initial_states, metadatas = batch
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = simulate_and_compute_mae(initial_states, metadatas, self.policy, self.itineraries, True, self.nb_traj_eval, 
                    self.pred_horizon, 'median', self.dataset_config['nb_future_station_reg'], self.device, self.sc, self.cat, self.stations_emb, self.lines_emb, self.dataset_config, self.delay_delta_bins, 
                    self.horizon_obs_bins, 'transformer')

        self.mae_delay_test += mae_delay
        self.mae_horizon_test += mae_horizon
        self.counter_delay_test += counter_delay
        self.counter_horizon_test += counter_horizon
        self.sse_test += sse
        
    def on_test_epoch_end(self) -> None:
        """
        Aggregate test metrics at the end of an epoch.

        Computes MAE per delay/horizon bin, overall MAE, MSE, and RMSE from
        accumulated values and stores them in `self.test_results`.

        Returns:
            None
        """
        sum_del_counter = torch.stack(self.counter_delay_test).sum(dim=0)
        sum_hor_counter = torch.stack(self.counter_horizon_test).sum(dim=0)
        
        mae_delay_tensor = torch.stack(self.mae_delay_test)
        counter_delay_tensor = torch.stack(self.counter_delay_test)
        
        mae_horizon_tensor = torch.stack(self.mae_horizon_test)
        counter_horizon_tensor = torch.stack(self.counter_horizon_test)
        
        mae_delay = (mae_delay_tensor * counter_delay_tensor).sum(dim=0) / sum_del_counter.clamp(min=1)
        mae_horizon = (mae_horizon_tensor * counter_horizon_tensor).sum(dim=0) / sum_hor_counter.clamp(min=1)
        mae = torch.dot(mae_horizon, sum_hor_counter) / sum_hor_counter.sum()

        mse = self.sse_test/ sum_hor_counter.sum()

        self.test_results = {
            "mae_horizon": mae_horizon.tolist(),
            "mae_delay": mae_delay.tolist(),
            "mae": mae.item(),
            "mse":mse.item(),
            "rmse":np.sqrt(mse).item(),
        }

    def on_train_start(self) -> None:
        """
        Pre-train discriminator and critic if using a pretrained BC policy.

        Returns:
            None
        """
        if self.uses_pretrained_bc:
            print("Pre-training discriminator:")
            max_rounds = 100
            for round_idx in range(1, max_rounds + 1):
                for _ in range(2):
                    self.update_disc()
                loss = self.evaluate_disc(max_batches=1)
                self.trainer.datamodule.update_policy_dataset(
                    nb_samples=self.sim_config['new_samples_per_epoch']
                )
                self.logger.log_metrics({'disc_loss': loss})
                if loss < 0.5:
                    break
            else:
                print(f"Warning: discriminator never reached target loss in {max_rounds} rounds")
        
            print("Pre-training critic:")
            best_loss = float('inf')
            patience_max = 15
            patience = 0
            max_rounds = 100
            for round_idx in range(1, max_rounds + 1):
                rewards = self.compute_rewards()
                values, final_values = self.compute_values()
                targets, gaes = self.compute_gae(rewards, values, final_values)
                loss = self.update_critic(targets, return_loss=True)
                self.trainer.datamodule.update_policy_dataset(
                    nb_samples=self.sim_config['new_samples_per_epoch']
                )
                self.logger.log_metrics({'critic_loss': loss})
                if loss < best_loss:
                    best_loss, patience = loss, 0
                else:
                    patience += 1
                    if patience >= patience_max:
                        print(f"No improvement for {patience} rounds; stopping critic pre-train.")
                        break
            else:
                print(f"Reached max {max_rounds} rounds without satisfying patience")

    def on_train_epoch_end(self) -> None:
        """
        Perform discriminator updates, PPO updates, and dataset refresh
        at the end of each training epoch.

        Returns:
            None
        """
        # The whole train epoch is done here
        loss = self.evaluate_disc(max_batches=1)
        print(loss)
        if loss > 0.45: # disc not good enough
            for _ in tqdm(range(self.disc_steps), desc="Disc steps: "):
                    self.update_disc()

        rewards = self.compute_rewards()
        values, final_values = self.compute_values()
        targets, gaes = self.compute_gae(rewards, values, final_values)

        for _ in tqdm(range(self.ppo_steps), desc="PPO steps: "):
            self.update_ppo(targets, gaes)
    
        self.update_parameters()

        self.trainer.datamodule.update_policy_dataset(nb_samples=self.sim_config['new_samples_per_epoch']) # Generate Data for the next epoch
        self.trainer.logger.log_epoch_metrics(self.trainer, self)

    def evaluate_disc(self, max_batches: int = None) -> float:
    """
    Evaluate the discriminator on policy and expert batches.

    Args:
        max_batches (int, optional): Maximum number of batches to evaluate.
            If None, use all batches. Default is None.

    Returns:
        float: Average discriminator loss over the evaluated batches.
    """
        self.discriminator.eval()
    
        losses_pi, losses_exp = [], []
    
        for i, (policy_batch, expert_batch) in enumerate(
                zip(self.trainer.datamodule.disc_policy_dataloader,
                    self.trainer.datamodule.disc_expert_dataloader)):
    
            if max_batches is not None and i >= max_batches:
                break
                
            x_pi, y_pi, mask_pi = [t.to(self.device) for t in policy_batch]
            x_exp, y_exp, mask_exp = [t.to(self.device) for t in expert_batch]

            logits_pi = self.discriminator(x_pi,  padding_mask=mask_pi).squeeze(-1)
            logits_exp = self.discriminator(x_exp, padding_mask=mask_exp).squeeze(-1)

            y_pi_smooth = y_pi.float() * (1.0 - self.smoothing_eps) + (1.0 - y_pi.float()) * self.smoothing_eps
            y_exp_smooth = y_exp.float() * (1.0 - self.smoothing_eps) + (1.0 - y_exp.float()) * self.smoothing_eps
    
            loss_pi = F.binary_cross_entropy_with_logits(logits_pi[~mask_pi], y_pi_smooth [~mask_pi])
            loss_exp = F.binary_cross_entropy_with_logits(logits_exp[~mask_exp], y_exp_smooth[~mask_exp])
    
            losses_pi.append(loss_pi)
            losses_exp.append(loss_exp)
            
        self.discriminator.train()
    
        loss_pi_mean = torch.stack(losses_pi).mean().item()
        loss_exp_mean = torch.stack(losses_exp).mean().item()
        loss_mean = loss_pi_mean + loss_exp_mean
    
        return loss_mean/2

    def update_disc(self) -> None:
        """
        Update the discriminator over paired policy/expert batches using epsilon smoothing and logs the losses.

        Returns:
            None
        """
        losses_pi = []
        losses_exp = []
        losses = []
        for policy_batch, expert_batch in zip(
                    self.trainer.datamodule.disc_policy_dataloader, 
                    self.trainer.datamodule.disc_expert_dataloader
                ): # Stops when every policy data is consumed
            x_pi, y_pi, mask_pi = [t.to(self.device) for t in policy_batch]
            x_exp, y_exp, mask_exp = [t.to(self.device) for t in expert_batch]
            
            logits_pi = self.discriminator(x_pi, padding_mask = mask_pi).squeeze(-1)
            logits_exp = self.discriminator(x_exp, padding_mask = mask_exp).squeeze(-1)

            with torch.no_grad():
                y_pi_smooth  = y_pi.float() * (1.0 - self.smoothing_eps) + (1.0 - y_pi .float()) * self.smoothing_eps
                y_exp_smooth = y_exp.float() * (1.0 - self.smoothing_eps) + (1.0 - y_exp.float()) * self.smoothing_eps

            loss_pi = F.binary_cross_entropy_with_logits(logits_pi[~mask_pi], y_pi_smooth[~mask_pi])
            loss_exp = F.binary_cross_entropy_with_logits(logits_exp[~mask_exp], y_exp_smooth[~mask_exp])
            loss = loss_pi + loss_exp
            
            optim_disc = self.trainer.optimizers[1]
            optim_disc.zero_grad()
            loss.backward()
            optim_disc.step()
            
            losses_pi.append(loss_pi.detach().cpu())
            losses_exp.append(loss_exp.detach().cpu())
            losses.append(loss.detach().cpu())

        self.log("disc_loss_pi", torch.stack(losses_pi).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("disc_loss_exp", torch.stack(losses_exp).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("disc_loss", torch.stack(losses).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        

    def update_ppo(self, targets: torch.Tensor, gaes: torch.Tensor) -> None:
        """
        Update both critic and policy using PPO.

        Args:
            targets (torch.Tensor): Value targets for critic update.
            gaes (torch.Tensor): Generalized advantage estimates for policy update.

        Returns:
            None
        """
        self.update_critic(targets)
        self.update_policy(gaes)

    def update_critic(self, targets: torch.Tensor, return_loss: bool = False) -> None | float:
        """
        Update the critic network with MSE loss against targets.

        Args:
            targets (torch.Tensor): Value targets for training the critic.
            return_loss (bool, optional): If True, return mean loss. Default is False.

        Returns:
            None or float: Mean critic loss if `return_loss` is True, otherwise None.
        """
        agents_ids = self.trainer.datamodule.policy_dataset.agents_ids
        losses_critic = []
        
        loss_fn = torch.nn.MSELoss()
        for states, _, _, _, idxs, padding_mask in self.trainer.datamodule.ppo_policy_dataloader:
            states = states.to(self.device)
            batch_targets = torch.cat([targets[idx][agents_ids[idx]] for idx in idxs])
            padding_mask = padding_mask.to(self.device)
            preds = self.critic(states, padding_mask = padding_mask).squeeze(-1)
            loss_critic = loss_fn(preds[~padding_mask], batch_targets)

            optim_critic = self.trainer.optimizers[2]
            optim_critic.zero_grad()
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            optim_critic.step()

            losses_critic.append(loss_critic.detach().cpu())
            
        loss = torch.stack(losses_critic).mean().item()

        self.log("loss_value", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        if return_loss:
            return loss

    def update_policy(self, adv: torch.Tensor) -> None:
        """
        Update the policy with PPO using provided advantages.

        Args:
            adv (torch.Tensor): Advantages aligned with sampled trajectories.

        Returns:
            None
        """
        agents_ids = self.trainer.datamodule.policy_dataset.agents_ids
        losses_actor = []
        entropies = []
        ratios_list = []
        advs_list = []
        dists_list = []
        
        for states, actions, old_probs, valid_actions_mask, idxs, padding_mask in self.trainer.datamodule.ppo_policy_dataloader:
            advs = torch.cat([adv[idx][agents_ids[idx]] for idx in idxs])
    
            states, actions, old_probs, valid_actions_mask, padding_mask = (
                tensor.to(self.device) for tensor in (states, actions, old_probs, valid_actions_mask, padding_mask)
            )

            logits = self.policy(states, padding_mask=padding_mask)
            logits = logits.masked_fill(~valid_actions_mask, float("-inf")) # set logits of invalid actions to -inf
            pi_dist = Categorical(logits=logits)
            
            new_log_prob = pi_dist.log_prob(actions.argmax(dim=-1)) # (B, L)
            old_log_prob = torch.log((old_probs * actions).sum(dim=-1)) # (B, L)) already masked the invalid actions at logit level in the environment

            mask = ~padding_mask
            ratios = (new_log_prob[mask] - old_log_prob[mask]).exp() # (B,L)
                
            loss_unclipped = -ratios * advs
            loss_clipped   = -torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
            loss_actor = torch.max(loss_unclipped, loss_clipped).mean()

            entropy_bonus = pi_dist.entropy()[mask].mean()

            total_loss = loss_actor - self.coef_ent * entropy_bonus

            optim_policy = self.trainer.optimizers[0]
            optim_policy.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            optim_policy.step()

            losses_actor.append(loss_actor.detach().cpu())
            entropies.append(entropy_bonus.detach().cpu())
            ratios_list.append(ratios.detach().cpu())
            advs_list.append(advs.detach().cpu())
            dists_list.append(pi_dist.probs[mask].detach().cpu())
        dists_mean = torch.cat(dists_list).mean(dim=0)

        self.log("same", dists_mean[0].item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("next1", dists_mean[1].item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("next2", dists_mean[2].item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("ratios_max", torch.cat(ratios_list).max(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("ratios_mean", torch.cat(ratios_list).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("advs_max", torch.cat(advs_list).max(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("advs_mean", torch.cat(advs_list).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("loss_actor", torch.stack(losses_actor).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("ent", torch.stack(entropies).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def compute_values(self) -> tuple:
        """
        Compute value predictions for rollout states using critic.

        Returns:
            tuple: (values, final_states_values), where each is a list of tensors.
        """
        self.critic.eval()
        values = []
        with torch.no_grad():
            for states,_,_,_,_,padding_mask in self.trainer.datamodule.ppo_policy_inf_dataloader:
                states = states.to(self.device)
                padding_mask = padding_mask.to(self.device)
                out = self.critic(states, padding_mask = padding_mask).squeeze(-1)
                values.extend([v[~mask] for v, mask in zip(out, padding_mask)])
            final_states, padding_mask = pad(self.trainer.datamodule.policy_dataset.final_states, device = self.device)
            final_states_values = self.critic(final_states, padding_mask = padding_mask).squeeze(-1)
            final_states_values = [v[~mask] for v, mask in zip(final_states_values, padding_mask)] 
        self.critic.train()
        return values, final_states_values

    def compute_gae(self, rewards_list: list, values_list: list, final_values_list: list) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Args:
            rewards_list (list): Per-step rewards (length B*T), each a 1D torch.Tensor of agent rewards.
            values_list (list): Per-step value predictions (length B*T), each a 1D torch.Tensor of agent values.
            final_values_list (list): Value predictions for final states per trajectory (length B), each a 1D torch.Tensor.

        Returns:
            tuple: (returns, advantages), both torch.Tensor of shape (T*B, N).
        """

        T = self.traj_len
        B = len(rewards_list) // T

        agents_ids = self.trainer.datamodule.policy_dataset.agents_ids
        final_agents_ids = self.trainer.datamodule.policy_dataset.final_agents_ids
        
        N = int(torch.cat(agents_ids + final_agents_ids).max().item()) + 1

        rewards = torch.zeros((B, T, N), device=self.device)
        values = torch.zeros((B, T, N), device=self.device)
        alive_now = torch.zeros((B, T, N), dtype=torch.bool, device=self.device)
        
        for i in range(B):
            for j in range(T):
                slot = i*T + j
                idx_now = agents_ids[slot]
        
                rewards[i, j, idx_now] = rewards_list[slot]
                values[i, j, idx_now] = values_list[slot]
                alive_now[i, j, idx_now] = True
        
        alive_next = torch.zeros_like(alive_now)
        alive_next[:, :-1] = alive_now[:, 1:]
        for i in range(B):
            alive_next[i, -1, final_agents_ids[i]] = True
        
        dones = (~alive_now) | (~alive_next) # alive now to avoid non zeros delta/gae before the beginning of the traj, alive_next to avoid error in delta/gae
        dones = dones.float()
        
        next_values = torch.zeros_like(values, device = self.device)
        next_values[:, :-1] = values[:, 1:]
        for i in range(B):
            next_values[i, -1, final_agents_ids[i]] = final_values_list[i]
    
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        
        gaes = torch.zeros_like(deltas)
        gae_acc = torch.zeros(B, N, device=self.device)
        for t in reversed(range(T)):
            gae_acc = deltas[:,t,:] + self.gamma * self.lambd * (1 - dones[:,t,:]) * gae_acc
            gaes[:,t,:] = gae_acc

        returns = gaes + values
        advantages = torch.zeros_like(gaes)
        relevant = (dones == 0)
        advantages[relevant] = (gaes[relevant] - gaes[relevant].mean()) / (gaes[relevant].std() + 1e-8)
        
        # reshape back to the dataset order
        return returns.view(T*B,N), advantages.view(T*B,N)

    def compute_rewards(self) -> list:
        """
        Compute rewards from the discriminator outputs for policy rollouts.

        Returns:
            list: Rewards as a list of 1D torch.Tensors, one per sequence element.
        """
        rewards = []
        self.discriminator.eval()
        with torch.no_grad():            
            for state_actions, _, padding_mask in self.trainer.datamodule.disc_policy_inf_dataloader:
                state_actions = state_actions.to(self.device)
                padding_mask = padding_mask.to(self.device)
                out = self.discriminator(state_actions, padding_mask = padding_mask).squeeze(-1)
                out = -F.logsigmoid(-out)
                rewards.extend([r[~mask] for r, mask in zip(out, padding_mask)])
                
        self.discriminator.train()
        
        self.log("rwd_mean", torch.cat(rewards).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("rwd_std", torch.cat(rewards).std(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
    
        return rewards

def pad(x: list, device: str = None) -> tuple:
    """
    Pad a list of variable-length tensors along the time dimension.

    Args:
        x (list): List of tensors of shape (seq_len, dim).
        device (str, optional): Device to place the result on. Default is None.

    Returns:
        tuple: (x_padded, padding_mask)
            x_padded (torch.Tensor): Tensor of shape (batch, max_len, dim).
            padding_mask (torch.Tensor): Bool mask of shape (batch, max_len), True for padding positions.
    """
    max_len = max(sequence.shape[0] for sequence in x)
    x_padded = torch.stack([torch.nn.functional.pad(sequence, (0, 0, 0, max_len - sequence.shape[0])) for sequence in x])
    
    padding_mask = torch.zeros(len(x), max_len, dtype=torch.bool)
    for i, sequence in enumerate(x):
        padding_mask[i, sequence.shape[0]:] = True

    if device is not None:
        x_padded = x_padded.to(device)
        padding_mask = padding_mask.to(device)

    return x_padded, padding_mask

def ppo_collate_fn(batch: list) -> tuple:
    """
    Collate function for PPO, padding variable-length sequences.

    Args:
        batch (list): List of tuples (x, y, logits, valid_actions_mask, idx).

    Returns:
        tuple: (x_padded, y_padded, logits_padded, valid_actions_mask_padded, idxs, padding_mask)
            x_padded (torch.Tensor): Input states, shape (batch, max_len, nb_feat).
            y_padded (torch.Tensor): Actions, shape (batch, max_len, nb_actions).
            logits_padded (torch.Tensor): Old action logits, shape (batch, max_len, nb_actions).
            valid_actions_mask_padded (torch.Tensor): Mask for valid actions, shape (batch, max_len, nb_actions).
            idxs (torch.Tensor): Indices of the original samples, used to keep track of which agent is at which tensor indices to compute advantages.
            padding_mask (torch.Tensor): Bool mask, shape (batch, max_len), True for padding.
    """
    x = [pair[0] for pair in batch]
    y = [pair[1] for pair in batch]
    logits = [pair[2] for pair in batch]
    valid_actions_mask = [pair[3] for pair in batch]
    idxs = torch.tensor([pair[4] for pair in batch])
    max_len = max(sequence.shape[0] for sequence in x)

    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    logits_padded = pad_sequence(logits, batch_first=True, padding_value=0)
    valid_actions_mask_padded = pad_sequence(valid_actions_mask, batch_first=True, padding_value=1)
    
    lengths = torch.tensor([seq.shape[0] for seq in x])
    padding_mask = torch.arange(x_padded.shape[1]) >= lengths.unsqueeze(1)
    
    return x_padded, y_padded, logits_padded, valid_actions_mask_padded, idxs, padding_mask

def disc_collate_fn(batch: list, mode: str) -> tuple:
    """
    Collate function for discriminator, padding sequences and assigning labels.

    Args:
        batch (list): List of tuples (x, y), where x and y are tensors of shape (seq_len, dim).
        mode (str): Source of data, must be "expert" or "policy".

    Returns:
        tuple: (state_action_tensor, labels, padding_mask)
            state_action_tensor (torch.Tensor): Concatenated [x, y], shape (batch, max_len, dim_x+dim_y).
            labels (torch.Tensor): Float labels, ones for expert and zeros for policy, shape (batch, max_len).
            padding_mask (torch.Tensor): Bool mask, shape (batch, max_len), True for padding.
    """

    x = [pair[0] for pair in batch]
    y = [pair[1] for pair in batch]
    max_len = max(sequence.shape[0] for sequence in x)

    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    state_action_tensor = torch.cat([x_padded, y_padded], dim=-1)
    
    lengths = torch.tensor([seq.shape[0] for seq in x])
    padding_mask = torch.arange(x_padded.shape[1]) >= lengths.unsqueeze(1)

    if mode == 'expert':
        labels = torch.ones((x_padded.shape[0], x_padded.shape[1]), dtype=torch.float)
    elif mode == 'policy':
        labels = torch.zeros((x_padded.shape[0], x_padded.shape[1]), dtype=torch.float)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return state_action_tensor, labels, padding_mask

class PolicyDataset(Dataset):
    """
    Dataset for storing trajectories collected by the policy.
    """

    def __init__(self) -> None:
        """
        Initialize empty buffers for policy data.
        """
        self.states = []
        self.final_states = []
        self.actions_one_hot = []
        self.actions_prob = []
        self.agents_ids = []
        self.final_agents_ids = []
        self.valid_actions_mask = [] 

    def __len__(self) -> int:
        """
        Number of stored samples.

        Returns:
            int: Dataset length.
        """
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve one stored sample.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (state, action_one_hot, action_prob, valid_actions_mask, idx).
        """
        return (self.states[idx],self.actions_one_hot[idx],self.actions_prob[idx],self.valid_actions_mask[idx],idx)

    def extend(self, states_list: list, one_hot_actions_list: list, prob_list: list,
               agents_ids: list, final_states: list, final_agent_ids: list, valid_actions_mask_list: list) -> None:
        """
        Extend the dataset with new trajectories.

        Args:
            states_list (list): List of state tensors.
            one_hot_actions_list (list): List of one-hot encoded actions.
            prob_list (list): List of action probability tensors.
            agents_ids (list): List of agent IDs per state.
            final_states (list): List of final states for trajectories.
            final_agent_ids (list): List of agent IDs for final states.
            valid_actions_mask_list (list): List of masks indicating valid actions.

        Returns:
            None
        """
        self.states += states_list
        self.final_states += final_states
        self.actions_one_hot += one_hot_actions_list
        self.actions_prob += prob_list
        self.agents_ids += agents_ids
        self.final_agents_ids += final_agent_ids
        self.valid_actions_mask += valid_actions_mask_list

    def clear(self) -> None:
        """
        Clear all stored data.

        Returns:
            None
        """
        self.states = []
        self.final_states = []
        self.actions_one_hot = []
        self.actions_prob = []
        self.agents_ids = []
        self.final_agents_ids = []
        self.valid_actions_mask = []

class InitialStateDataset(Dataset):
    """
    Dataset wrapper for tensor loading of initial states used in the simulation.

    Args:
        base_path (str): Root directory of the dataset.
        config (dict): Config dict containing split sizes (e.g. "train_size").
        scheme (dict): Data scheme.
        ratio (float): Fraction of the split to keep (0–1).
        split (str): Dataset split ("train", "val", "test").
    """
    def __init__(self, base_path: str, config: dict, scheme: dict, ratio: float, split: str) -> None:
        self.kept_cols = scheme['cols_to_keep']
        self.ratio = ratio
        total_data = config[f"{split}_size"]
        self.x_path = os.path.join(base_path, split, 'x')
        self.y_path = os.path.join(base_path, split, 'y_actions')
        self.md_path = os.path.join(base_path, split, 'md')
        self.mapper = torch.linspace(0, total_data - 1, int(total_data*ratio), dtype=int)
        self.len = self.mapper.shape[0]

    def __len__(self) -> int:
        """
        Dataset size.

        Returns:
            int: Number of samples.
        """
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        """
        Load one sample from disk.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (x, y, md) where
                x (torch.Tensor): Input features with selected columns.
                y (torch.Tensor): Target actions.
                md (torch.Tensor): Metadata for the sample.
        """
        idx = self.mapper[idx]
        x = torch.load(get_subdir_path(f'x_{idx}.pt',self.x_path), weights_only = False)[:, self.kept_cols]
        y = torch.load(get_subdir_path(f'y_actions_{idx}.pt',self.y_path), weights_only = False)
        md = torch.load(get_subdir_path(f'md_{idx}.pt',self.md_path), weights_only = False)
        return x, y, md

class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading train/val/test datasets.

    Args:
        data_path (str): Root path to the dataset.
        dataset_config (dict): Dataset configuration (e.g., split sizes).
        sim_config (dict): Simulation configuration (trajectories length, simulation batch size, ...).
        scheme (dict): Data scheme.
        cat (dict): Categorical feature definitions.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        batch_size (int): Batch size.
        num_workers (int, optional): DataLoader workers. Default is 0.
        pin_memory (bool, optional): Whether to pin memory in DataLoader. Default is True.
        eval_test (bool, optional): If True, use test split for evaluation. Default is False.
        val_ratio (float, optional): Fraction of validation data to keep. Default is 1.0.
    """
    def __init__(self, data_path: str, dataset_config: dict, sim_config: dict, scheme: dict, cat: dict, stations_emb: dict, lines_emb: dict, batch_size: int, 
                num_workers=0, pin_memory=True, eval_test=False, val_ratio=1.0) -> None:
        super().__init__()
        self.base_path = data_path
        self.dataset_config = dataset_config
        self.scheme = scheme
        self.cat = cat
        self.stations_emb = stations_emb
        self.lines_emb = lines_emb
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sim_config = sim_config
        self.eval_test = eval_test
        self.val_ratio = val_ratio

    def setup(self, stage: str) -> None:
        """
        Set up datasets and dataloaders for the given stage. If stage is "fit", perform the initial filling of the replay buffer and initial the dataloaders for GAIL training loop.

        Args:
            stage (str): Current stage ("fit", "validate", "test", "predict").

        Returns:
            None
        """
        self.policy_dataset = PolicyDataset()

        self.expert_dataset = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, 1.0, 'train')
        if self.eval_test:
            self.init_states_val_ds = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, 1.0, 'val')
            self.expert_dataset = ConcatDataset([self.expert_dataset, self.init_states_val_ds])
            self.init_states_test_ds  = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, 1.0, 'test')
        else:
            self.init_states_val_ds = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, self.val_ratio, 'val')
            
        self.initial_state_dataloader = self.get_initial_state_dataloader()
        
        if stage == "fit":
            print('Initial policy dataset filling.')
            self.update_policy_dataset(nb_samples=self.sim_config['new_samples_per_epoch']) # initial fill of the replay buffer
            print('Filled !')
            self.disc_policy_dataloader = self.get_disc_policy_dataloader()
            self.disc_policy_inf_dataloader = self.get_disc_policy_inference_dataloader()
            self.disc_expert_dataloader = self.get_disc_expert_dataloader()
            self.ppo_policy_dataloader = self.get_ppo_policy_dataloader()
            self.ppo_policy_inf_dataloader = self.get_ppo_policy_inference_dataloader()
        
    def train_dataloader(self) -> None:
        """ 
        Dummy dataloader.
        """
        dummy_data = [0]
        return DataLoader(dummy_data, batch_size=1)

    def val_dataloader(self) -> None:
        """ 
        Dummy dataloader.
        """
        dummy_data = [0]
        return DataLoader(dummy_data, batch_size=1)
    
    def get_initial_state_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for initial states.

        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return DataLoader(
            self.expert_dataset,
            batch_size=self.sim_config['sim_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda x: [(el[0], el[2]) for el in x] # stacking with padding is done within the simulator
        )

    def get_disc_policy_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for discriminator training with policy data.

        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return DataLoader(
            self.policy_dataset,
            batch_size=int(self.batch_size/2), # Because batches are merged with expert data
            num_workers=int(self.num_workers/2),
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=partial(disc_collate_fn, mode='policy')
        )

    def get_disc_policy_inference_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for discriminator inference with policy data.

        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return DataLoader(
            self.policy_dataset,
            batch_size=256, # Because batches are merged with expert data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=partial(disc_collate_fn, mode='policy')
        )

    def get_disc_expert_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for discriminator training with expert data.

        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return DataLoader(
            self.expert_dataset,
            batch_size=int(self.batch_size/2), # Because batches are merged with policy data
            num_workers=int(self.num_workers/2), # Because batches are merged with policy data
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=partial(disc_collate_fn, mode='expert')
        )

    def get_ppo_policy_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for ppo training with policy data.

        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return DataLoader(
            self.policy_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=ppo_collate_fn
        )   

    def get_ppo_policy_inference_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for policy inference with policy data.

        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return DataLoader(
            self.policy_dataset,
            batch_size=256,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=ppo_collate_fn
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for validation set.

        Returns:
            torch.utils.data.DataLoader or list: Validation data loader,
            or an empty list if `eval_test` is True.
        """
        if self.eval_test:
            return []
        return DataLoader(
            self.init_states_val_ds,
            batch_size=2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: ([el[0] for el in x], [el[2] for el in x])
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader or list:
        """
        Create DataLoader for test set.

        Returns:
            torch.utils.data.DataLoader or list: Test data loader
            or an empty list if `eval_test` is False
        """
        if self.eval_test:
            return DataLoader(
                self.init_states_test_ds,
                batch_size=2,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=lambda x: ([el[0] for el in x], [el[2] for el in x])
            )
        else:
            return []            

    def update_policy_dataset(self, nb_samples: int) -> None:
        """
        Populate the policy dataset with simulated rollouts.

        Args:
            nb_samples (int): Total number of samples to collect; must be divisible by traj_len * sim_batch_size.

        Returns:
            None
        """
        self.policy_dataset.clear()
        self.trainer.model.policy.eval()

        sim = Simulator(self.trainer.model.policy, 
                        self.dataset_config['deltat'], 
                        self.scheme['x'],
                        self.cat,
                        self.stations_emb, 
                        self.lines_emb,
                        self.sim_config['device'], 
                        self.dataset_config['nb_past_station_sim'], 
                        self.dataset_config['nb_future_station_sim'], 
                        self.dataset_config['embedding_size'], 
                        self.dataset_config['idle_end'],
                       'transformer')
        
        if nb_samples % (self.sim_config['traj_len'] * self.sim_config['sim_batch_size']) != 0:
            raise ValueError("nb_samples must be divisible by traj_len * sim_batch_size")

        nb_collected_samples = 0
        for batch in self.initial_state_dataloader:
            # maybe cut the batch here when needed to avoid additional computing ?
            initial_states = [el[0] for el in batch]
            metadatas = [el[1] for el in batch]
            states_time = [metadatas[i][0,0] for i in range(self.sim_config['sim_batch_size'])]
            initial_states_metadata = [metadatas[i][:,1:] for i in range(self.sim_config['sim_batch_size'])]
        
            with torch.no_grad():
                states_list, one_hot_actions_list, prob_list, ids_list, final_states, final_ids, valid_actions_mask_list = sim.get_samples_gail(initial_states, initial_states_metadata, states_time, self.sim_config['traj_len'], 1,'sampling', False, itineraries = self.sim_config['itineraries'])
                
            self.policy_dataset.extend(states_list, one_hot_actions_list, prob_list, ids_list, final_states, final_ids, valid_actions_mask_list)
            
            sim.reset()
            del batch, initial_states, metadatas, states_time, initial_states_metadata
            gc.collect()
            torch.cuda.empty_cache()
            
            nb_collected_samples += len(states_list) 
            if nb_collected_samples == nb_samples:
                break
        
        self.trainer.model.policy.train()

def main():
    """
    Entry point for training and evaluation.

    Parses CLI arguments, sets up data, model, logger, callbacks, and
    launches training (and testing if `--eval-test` is set).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="Name of the experiment.")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("itineraries_path", type=str, help="Path to the itineraries.")
    parser.add_argument("eval_config_path", type=str, help="Path to the evaluation config.")
    parser.add_argument("d_model", type=int, help="Dimensionality of the transformer embeddings (model hidden size).")
    parser.add_argument("nhead", type=int, help="Number of attention heads in each multi-head self-attention layer.")
    parser.add_argument("dim_feedforward", type=int, help="Hidden layer size of the position-wise feedforward sublayer.")
    parser.add_argument("dropout", type=float, help="Dropout probability for all dropout layers (0.0–1.0).")
    parser.add_argument("activation", type=str, choices=["relu","gelu","tanh"], help="Activation function to use in the feedforward layers.")
    parser.add_argument("num_layers", type=int, help="Number of stacked TransformerEncoder layers.")
    parser.add_argument("traj_len", type=int, help="Number of steps per trajectory.")
    parser.add_argument("sim_batch_size", type=int, help="Number of simulations to do in parallel.")
    parser.add_argument("new_samples_per_epoch", type=int, help="Number of new samples to generate per epoch.")
    parser.add_argument("nb_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("disc_steps", type=int, help="Number of discriminator steps each epoch.")
    parser.add_argument("ppo_steps", type=int, help="Number of PPO steps each epoch.")
    parser.add_argument("smoothing_eps", type=float, metavar="ε", help="Label-smoothing factor ε for the discriminator (0 ⇒ no smoothing).")
    parser.add_argument("entropy_max",   type=float, help="Initial / maximum entropy coefficient before decay (β_start).")
    parser.add_argument("entropy_min",   type=float, help="Minimum entropy coefficient after decay (β_end).")
    parser.add_argument("batch_size", type=int, help="Training batch size.")
    parser.add_argument("lr_p", type=float, help="Policy LR for Adam")
    parser.add_argument("weight_decay_p", type=float, help="Policy AdamW L2")
    parser.add_argument("lr_d", type=float, help="Discriminator LR for Adam")
    parser.add_argument("weight_decay_d", type=float, help="Discriminator AdamW L2")
    parser.add_argument("lr_c", type=float, help="Critic LR for Adam")
    parser.add_argument("weight_decay_c", type=float, help="Critic AdamW L2")
    parser.add_argument("check_val_every_n_epoch", type=int, help="Number of epochs between each validation check.")
    parser.add_argument("num_workers", type=int, help="Number of subprocesses to use for data loading.")
    parser.add_argument("min_epochs", type=int, help="Minimum number of epochs before early stopping can trigger.")
    parser.add_argument("patience", type=int, help="Number of validation checks with no improvement before early stopping triggers.")
    parser.add_argument("--val-ratio", type=float, default=1.0, help="Ratio of validation data kept to compute metrics during training.")
    parser.add_argument("--eval-test", action="store_true", help="If true, train on train+val and evaluate on test, if false, train on train and evaluate on val.")
    parser.add_argument("--bc-pretrained", action="store_true", help="If set, loads BC pre_trained policy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    print(args)

    seed_everything(args.seed, workers=True)

    scheme = load_pickle(os.path.join(args.data_path, 'sc_sim_non.pkl'))
    cat = load_pickle(os.path.join(args.data_path, 'cat.pkl'))
    stations_emb = load_pickle(os.path.join(args.data_path, 'stations_emb.pkl'))
    lines_emb = load_pickle(os.path.join(args.data_path, 'lines_emb.pkl'))
    dataset_config = load_pickle(os.path.join(args.data_path, 'config.pkl'))

    model_config = {
        "input_dim":len(scheme['x'].items()),
        "d_model":args.d_model,
        "nhead":args.nhead,
        "dim_feedforward":args.dim_feedforward,
        "dropout":args.dropout,
        "activation":args.activation,
        "num_layers":args.num_layers,
        "num_classes":len(scheme['y'].items())
    }

    eval_months = ['train','val','test']
    dates = get_dates(args.data_path, eval_months)
    itineraries = load_itineraries_from_dates(dates, args.itineraries_path, show_prog = True)

    sim_config = {
        'traj_len':args.traj_len,
        'sim_batch_size':args.sim_batch_size,
        'new_samples_per_epoch':args.new_samples_per_epoch,
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'itineraries':itineraries
    }

    eval_config = load_pickle(args.eval_config_path)
    
    model = GAIL(model_config, dataset_config, sim_config, args.lr_p, args.weight_decay_p, args.lr_d, args.weight_decay_d, args.lr_c, args.weight_decay_c, args.disc_steps, args.ppo_steps, 
                args.smoothing_eps, args.entropy_max, args.entropy_min, args.bc_pretrained, itineraries, scheme['x'], cat, stations_emb, lines_emb, eval_config)

    if args.bc_pretrained:
        pre_trained_path = "Runs/sim_class/run_input_dim234_d_model512_nhead4_dim_feedforward2048_dropout0.2_activationrelu_num_layers4_num_classes3"
        state_dict = torch.load(os.path.join(pre_trained_path,"checkpoints", f"model_epoch_1.pt"), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # put whatever pretrained checkpoint you want
        gail.policy.load_state_dict(state_dict)
    
    run_path, checkpoints_path = setup_run_folder(args, model_config, 'tr_gail')
    
    data_module = DataModule(args.data_path, dataset_config, sim_config, scheme, cat, stations_emb, lines_emb, args.batch_size, num_workers=args.num_workers, eval_test=args.eval_test, 
                val_ratio=args.val_ratio)

    logger = CustomCSVLogger(save_dir=run_path, 
                             train_metrics=["disc_loss","disc_loss_pi","disc_loss_exp","same","next1","next2","ratios_max","ratios_mean","advs_max","advs_mean","loss_actor","loss_value","ent",
                                        "rwd_mean","rwd_std"],
                             val_metrics=[f"hor{i}" for i in range(len(eval_config['horizon_obs_bins'])-1)] + 
                                         [f"del{i}" for i in range(len(eval_config['delay_delta_bins'])-1)] + 
                                         ["mae"])

    callbacks = [
        WeightSaver(save_path=checkpoints_path, 
                    target="policy", 
                    keep_module_prefix=False,
                    top_k=5,
                    monitor="mae",
                    mode="min"),
        MetricsLoggingCallback(logger)
    ]

    if not args.eval_test:
        early_stop = EarlyStopping(
            monitor="mae",
            mode="min",
            patience=args.patience
        )
        callbacks.append(early_stop)
    
    trainer = Trainer(
        devices=1,
        strategy="auto",
        accelerator="gpu",
        max_epochs=args.nb_epochs,
        logger=logger,
        check_val_every_n_epoch =args.check_val_every_n_epoch,
    	callbacks=callbacks,
        enable_checkpointing=False,
        num_sanity_val_steps = 0
    )
    
    trainer.fit(model, data_module)

    if args.eval_test:
        trainer.test(model, datamodule=data_module)
        print(model.test_results)

if __name__ == "__main__":
    main()