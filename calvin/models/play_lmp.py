from collections import OrderedDict
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch
import torch.distributed as dist
import torch.distributions as D
from torch.distributions.distribution import Distribution
import torch.nn as nn
from torch.nn.functional import mse_loss

from calvin.models.decoders.action_decoder import ActionDecoder
from calvin.utils.visualizations import visualize_temporal_consistency

logger = logging.getLogger(__name__)


class PlayLMP(pl.LightningModule):
    def __init__(
        self,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        vision_static: DictConfig,
        visual_goal: DictConfig,
        language_goal: DictConfig,
        state_decoder: DictConfig,
        decoder: DictConfig,
        proprio_state: DictConfig,
        vision_gripper: Optional[DictConfig],
        kl_beta: float,
        state_recon_beta: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        state_reconstruction: bool,
    ):
        super(PlayLMP, self).__init__()
        self.state_recons = state_reconstruction
        self.setup_input_sizes(
            vision_static,
            vision_gripper,
            plan_proposal,
            plan_recognition,
            visual_goal,
            decoder,
            state_decoder,
            proprio_state,
        )

        self.plan_proposal = hydra.utils.instantiate(plan_proposal)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition)
        self.vision_static = hydra.utils.instantiate(vision_static)
        self.vision_gripper = hydra.utils.instantiate(vision_gripper) if vision_gripper else None
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(decoder)
        if self.state_recons:
            self.state_decoder = hydra.utils.instantiate(state_decoder)
        self.kl_beta = kl_beta
        self.st_recon_beta = state_recon_beta
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        # workaround to resolve hydra config file before calling save_hyperparams  until they fix this issue upstream
        # without this, there is conflict between lightning and hydra
        decoder.out_features = decoder.out_features

        self.optimizer_config["lr"] = self.optimizer_config["lr"]
        self.save_hyperparameters()

    @staticmethod
    def setup_input_sizes(
        vision_static,
        vision_gripper,
        plan_proposal,
        plan_recognition,
        visual_goal,
        decoder,
        state_decoder,
        proprio_state,
    ):
        # remove a dimension if we convert robot orientation quaternion to euler angles
        n_state_obs = int(np.sum(np.diff([list(x) for x in [list(y) for y in proprio_state.keep_indices]])))

        plan_proposal.n_state_obs = n_state_obs
        plan_recognition.n_state_obs = n_state_obs
        visual_goal.n_state_obs = n_state_obs
        decoder.n_state_obs = n_state_obs
        state_decoder.n_state_obs = n_state_obs

        visual_features = vision_static.visual_features
        if vision_gripper:
            visual_features += vision_gripper.visual_features

        plan_proposal.visual_features = visual_features
        plan_recognition.visual_features = visual_features
        visual_goal.visual_features = visual_features
        decoder.visual_features = visual_features

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        assert isinstance(self.trainer, pl.Trainer)
        combined_loader_dict = self.trainer.datamodule.train_dataloader()  # type: ignore
        dataset_lengths = [len(combined_loader_dict[k]) for k in combined_loader_dict.keys()]
        dataset_size = max(dataset_lengths)
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices  # type: ignore
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs  # type: ignore

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:  # type: ignore
            return self.trainer.max_steps  # type: ignore
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        num_warmup_steps = int(num_warmup_steps)
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if "num_warmup_steps" in self.lr_scheduler:
            self.lr_scheduler.num_training_steps, self.lr_scheduler.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.lr_scheduler.num_training_steps,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.lr_scheduler.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.lr_scheduler.num_warmup_steps}")
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def visual_embedding(self, imgs: List[torch.Tensor]) -> torch.Tensor:  # type: ignore
        if len(imgs) != 2:
            imgs_static, imgs_gripper = imgs[0], None
        else:
            imgs_static, imgs_gripper = imgs
        b, s, c, h, w = imgs_static.shape
        imgs_static = imgs_static.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 300, 300)
        # ------------ Vision Network ------------ #
        encoded_imgs = self.vision_static(imgs_static)  # (batch*seq_len, 64)
        encoded_imgs = encoded_imgs.reshape(b, s, -1)  # (batch, seq, 64)

        if imgs_gripper is not None:
            b, s, c, h, w = imgs_gripper.shape
            imgs_gripper = imgs_gripper.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 300, 300)
            # ------------ Vision Network ------------ #
            encoded_imgs_gripper = self.vision_gripper(imgs_gripper)  # (batch*seq_len, 64)
            encoded_imgs_gripper = encoded_imgs_gripper.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_gripper], dim=-1)
        return encoded_imgs

    def perceptual_embedding(self, visual_emb: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:  # type: ignore
        perceptual_emb = torch.cat([visual_emb, obs], dim=-1)
        return perceptual_emb

    def lmp_train(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, train_acts: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.distributions.Distribution
    ]:
        # ------------Plan Proposal------------ #
        pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each

        # ------------Plan Recognition------------ #
        pr_dist = self.plan_recognition(perceptual_emb)  # (batch, 256) each

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        action_loss = self.action_decoder.loss(sampled_plan, perceptual_emb, latent_goal, train_acts)
        kl_loss = self.compute_kl_loss(pr_dist, pp_dist)
        total_loss = action_loss + kl_loss

        return kl_loss, action_loss, total_loss, pp_dist, pr_dist

    def lmp_val(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # ------------Plan Proposal------------ #
        pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each

        # ------------ Policy network ------------ #
        sampled_plan_pp = pp_dist.sample()  # sample from proposal net
        action_loss_pp, sample_act_pp = self.action_decoder.loss_and_act(
            sampled_plan_pp, perceptual_emb, latent_goal, actions
        )

        mae_pp = torch.nn.functional.l1_loss(
            sample_act_pp[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pp = torch.mean(mae_pp, 1)  # (batch, 6)
        # gripper action
        gripper_discrete_pp = sample_act_pp[..., -1]
        gt_gripper_act = actions[..., -1]
        m = gripper_discrete_pp > 0
        gripper_discrete_pp[m] = 1
        gripper_discrete_pp[~m] = -1
        gripper_sr_pp = torch.mean((gt_gripper_act == gripper_discrete_pp).float())

        # ------------Plan Recognition------------ #
        pr_dist = self.plan_recognition(perceptual_emb)  # (batch, 256) each

        sampled_plan_pr = pr_dist.sample()  # sample from recognition net
        action_loss_pr, sample_act_pr = self.action_decoder.loss_and_act(
            sampled_plan_pr, perceptual_emb, latent_goal, actions
        )
        mae_pr = torch.nn.functional.l1_loss(
            sample_act_pr[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pr = torch.mean(mae_pr, 1)  # (batch, 6)
        kl_loss = self.compute_kl_loss(pr_dist, pp_dist)
        # gripper action
        gripper_discrete_pr = sample_act_pr[..., -1]
        m = gripper_discrete_pr > 0
        gripper_discrete_pr[m] = 1
        gripper_discrete_pr[~m] = -1
        gripper_sr_pr = torch.mean((gt_gripper_act == gripper_discrete_pr).float())

        return (
            sampled_plan_pp,
            action_loss_pp,
            sampled_plan_pr,
            action_loss_pr,
            kl_loss,
            mae_pp,
            mae_pr,
            gripper_sr_pp,
            gripper_sr_pr,
        )

    def training_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Tuple[
                torch.Tensor,
                List[torch.Tensor],
                List[torch.Tensor],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        batch_idx: int,
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        batch: list( batch_dataset_vision, batch_dataset_lang, ..., batch_dataset_differentModalities)
            - batch_dataset_vision: tuple( train_obs: Tensor,
                                           train_rgbs: tuple(Tensor, ),
                                           train_depths: tuple(Tensor, ),
                                           train_acts: Tensor ),
                                           info: Dict,
                                           idx: int
            - batch_dataset_lang: tuple( train_obs: Tensor,
                                         train_rgbs: tuple(Tensor, ),
                                         train_depths: tuple(Tensor, ),
                                         train_acts: Tensor,
                                         train_lang: Tensor   ),
                                         info: Dict,
                                         idx: int
        """
        kl_loss, action_loss, proprio_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        encoders_dict = {}
        for self.modality_scope, dataset_batch in batch.items():
            batch_obs, batch_rgbs, batch_depths, batch_acts, batch_encoded_lang, _, idx = dataset_batch
            visual_emb = self.visual_embedding(batch_rgbs)
            if self.state_recons:
                proprio_pred = self.state_decoder(visual_emb)
                p_loss = mse_loss(batch_obs, proprio_pred)
                proprio_loss += p_loss
            perceptual_emb = self.perceptual_embedding(visual_emb, batch_obs)
            latent_goal = (
                self.visual_goal(perceptual_emb[:, -1])
                if "vis" in self.modality_scope
                else self.language_goal(batch_encoded_lang)
            )
            kl, act_loss, mod_loss, pp_dist, pr_dist = self.lmp_train(perceptual_emb, latent_goal, batch_acts)
            encoders_dict[self.modality_scope] = [pp_dist, pr_dist]
            kl_loss += kl
            action_loss += act_loss
            total_loss += mod_loss
            self.log(f"train/action_loss_{self.modality_scope}", act_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"train/total_loss_{self.modality_scope}", mod_loss, on_step=False, on_epoch=True, sync_dist=True)
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        kl_loss = kl_loss / len(batch)
        action_loss = action_loss / len(batch)
        proprio_loss = proprio_loss / len(batch)
        total_loss = total_loss + self.st_recon_beta * proprio_loss if self.state_recons else total_loss
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/pred_proprio", self.st_recon_beta * proprio_loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": total_loss, "encoders_dict": encoders_dict}

    def compute_kl_loss(
        self, pr_dist: torch.distributions.Distribution, pp_dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        kl_loss = D.kl_divergence(pr_dist, pp_dist).mean()
        kl_loss_scaled = kl_loss * self.kl_beta
        self.log(f"train/kl_loss_{self.modality_scope}", kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            f"train/kl_loss_scaled_{self.modality_scope}", kl_loss_scaled, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log("train/kl_beta", self.kl_beta, on_step=False, on_epoch=True, sync_dist=True)
        return kl_loss_scaled

    def set_kl_beta(self, kl_beta):
        """Set kl_beta from Callback"""
        self.kl_beta = kl_beta

    def validation_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Tuple[
                torch.Tensor,
                List[torch.Tensor],
                List[torch.Tensor],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        batch: list( batch_dataset_vision, batch_dataset_lang, ..., batch_dataset_differentModalities)
            - batch_dataset_vision: tuple( train_obs: Tensor,
                                           train_rgbs: tuple(Tensor, ),
                                           train_depths: tuple(Tensor, ),
                                           train_acts: Tensor ),
                                           info: Dict,
                                           idx: int
            - batch_dataset_lang: tuple( train_obs: Tensor,
                                         train_rgbs: tuple(Tensor, ),
                                         train_depths: tuple(Tensor, ),
                                         train_acts: Tensor,
                                         train_lang: Tensor   ),
                                         info: Dict,
                                         idx: int
        """
        output = {}
        for self.modality_scope, dataset_batch in batch.items():
            batch_obs, batch_rgbs, batch_depths, batch_acts, batch_encoded_lang, _, idx = dataset_batch
            visual_emb = self.visual_embedding(batch_rgbs)
            if self.state_recons:
                proprio_pred = self.state_decoder(visual_emb)
                p_loss = mse_loss(batch_obs, proprio_pred)
                output[f"proprio_pred_{self.modality_scope}"] = p_loss
            perceptual_emb = self.perceptual_embedding(visual_emb, batch_obs)
            latent_goal = (
                self.visual_goal(perceptual_emb[:, -1])
                if "vis" in self.modality_scope
                else self.language_goal(batch_encoded_lang)
            )
            (
                sampled_plan_pp,
                action_loss_pp,
                sampled_plan_pr,
                action_loss_pr,
                kl_loss,
                mae_pp,
                mae_pr,
                gripper_sr_pp,
                gripper_sr_pr,
            ) = self.lmp_val(perceptual_emb, latent_goal, batch_acts)
            output[f"val_action_loss_pp_{self.modality_scope}"] = action_loss_pp
            output[f"sampled_plan_pp_{self.modality_scope}"] = sampled_plan_pp
            output[f"val_action_loss_pr_{self.modality_scope}"] = action_loss_pr
            output[f"sampled_plan_pr_{self.modality_scope}"] = sampled_plan_pr
            output[f"kl_loss_{self.modality_scope}"] = kl_loss
            output[f"mae_pp_{self.modality_scope}"] = mae_pp
            output[f"mae_pr_{self.modality_scope}"] = mae_pr
            output[f"gripper_sr_pp{self.modality_scope}"] = gripper_sr_pp
            output[f"gripper_sr_pr{self.modality_scope}"] = gripper_sr_pr
            output[f"idx_{self.modality_scope}"] = idx

        return output

    def validation_epoch_end(self, validation_step_outputs):
        val_total_act_loss_pr = torch.tensor(0.0).to(self.device)
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        val_kl_loss = torch.tensor(0.0).to(self.device)
        val_total_proprio_loss = torch.tensor(0.0).to(self.device)
        val_total_mae_pr = torch.tensor(0.0).to(self.device)
        val_total_mae_pp = torch.tensor(0.0).to(self.device)
        val_pos_mae_pp = torch.tensor(0.0).to(self.device)
        val_pos_mae_pr = torch.tensor(0.0).to(self.device)
        val_orn_mae_pp = torch.tensor(0.0).to(self.device)
        val_orn_mae_pr = torch.tensor(0.0).to(self.device)
        val_grip_sr_pr = torch.tensor(0.0).to(self.device)
        val_grip_sr_pp = torch.tensor(0.0).to(self.device)
        for mod in self.trainer.datamodule.modalities:
            act_loss_pp = torch.stack([x[f"val_action_loss_pp_{mod}"] for x in validation_step_outputs]).mean()
            act_loss_pr = torch.stack([x[f"val_action_loss_pr_{mod}"] for x in validation_step_outputs]).mean()
            kl_loss = torch.stack([x[f"kl_loss_{mod}"] for x in validation_step_outputs]).mean()
            mae_pp = torch.cat([x[f"mae_pp_{mod}"] for x in validation_step_outputs])
            mae_pr = torch.cat([x[f"mae_pr_{mod}"] for x in validation_step_outputs])
            pr_mae_mean = mae_pr.mean()
            pp_mae_mean = mae_pp.mean()
            pos_mae_pp = mae_pp[..., :3].mean()
            pos_mae_pr = mae_pr[..., :3].mean()
            orn_mae_pp = mae_pp[..., 3:6].mean()
            orn_mae_pr = mae_pr[..., 3:6].mean()
            grip_sr_pp = torch.stack([x[f"gripper_sr_pp{mod}"] for x in validation_step_outputs]).mean()
            grip_sr_pr = torch.stack([x[f"gripper_sr_pr{mod}"] for x in validation_step_outputs]).mean()
            val_total_mae_pr += pr_mae_mean
            val_total_mae_pp += pp_mae_mean
            val_pos_mae_pp += pos_mae_pp
            val_pos_mae_pr += pos_mae_pr
            val_orn_mae_pp += orn_mae_pp
            val_orn_mae_pr += orn_mae_pr
            val_grip_sr_pp += grip_sr_pp
            val_grip_sr_pr += grip_sr_pr
            val_total_act_loss_pp += act_loss_pp
            val_total_act_loss_pr += act_loss_pr
            val_kl_loss += kl_loss
            if self.state_recons:
                proprio_loss = torch.stack([x[f"proprio_pred_{mod}"] for x in validation_step_outputs]).mean()
                val_total_proprio_loss += proprio_loss
            self.log(f"val_act/{mod}_act_loss_pp", act_loss_pp, sync_dist=True)
            self.log(f"val_act/{mod}_act_loss_pr", act_loss_pr, sync_dist=True)
            self.log(f"val_total_mae/{mod}_total_mae_pr", pr_mae_mean, sync_dist=True)
            self.log(f"val_total_mae/{mod}_total_mae_pp", pp_mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{mod}_pos_mae_pr", pos_mae_pr, sync_dist=True)
            self.log(f"val_pos_mae/{mod}_pos_mae_pp", pos_mae_pp, sync_dist=True)
            self.log(f"val_orn_mae/{mod}_orn_mae_pr", orn_mae_pr, sync_dist=True)
            self.log(f"val_orn_mae/{mod}_orn_mae_pp", orn_mae_pp, sync_dist=True)
            self.log(f"val_grip/{mod}_grip_sr_pr", grip_sr_pr, sync_dist=True)
            self.log(f"val_grip/{mod}_grip_sr_pp", grip_sr_pp, sync_dist=True)
            self.log(f"val_kl/{mod}_kl_loss", kl_loss, sync_dist=True)
        self.log(
            "val_act/action_loss_pp", val_total_act_loss_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log(
            "val_act/action_loss_pr", val_total_act_loss_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log("val_kl/kl_loss", val_kl_loss / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log(
            "val_total_mae/total_mae_pr", val_total_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log(
            "val_total_mae/total_mae_pp", val_total_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log("val_pos_mae/pos_mae_pr", val_pos_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_pos_mae/pos_mae_pp", val_pos_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_orn_mae/orn_mae_pr", val_orn_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_orn_mae/orn_mae_pp", val_orn_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_grip/grip_sr_pr", val_grip_sr_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_grip/grip_sr_pp", val_grip_sr_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        if self.state_recons:
            self.log(
                "val/proprio_loss", val_total_proprio_loss / len(self.trainer.datamodule.modalities), sync_dist=True
            )

    def predict_with_plan(
        self,
        curr_imgs: List[torch.Tensor],
        curr_state: torch.Tensor,
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:

        imgs = [curr_img.unsqueeze(0) for curr_img in curr_imgs]
        curr_state = curr_state.unsqueeze(0)

        with torch.no_grad():
            visual_emb = self.visual_embedding(imgs)
            perceptual_emb = self.perceptual_embedding(visual_emb, curr_state)
            action = self.action_decoder.act(sampled_plan, perceptual_emb, latent_goal)

        return action

    def get_pp_plan_vision(
        self,
        curr_imgs: List[torch.Tensor],
        goal_imgs: List[torch.Tensor],
        curr_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(curr_imgs) == len(goal_imgs)
        imgs = [
            torch.cat([curr_img, goal_img]).unsqueeze(0) for curr_img, goal_img in zip(curr_imgs, goal_imgs)
        ]  # (1, 2, C, H, W)
        state = torch.cat((curr_state, goal_state)).unsqueeze(0)
        with torch.no_grad():
            visual_emb = self.visual_embedding(imgs)
            perceptual_emb = self.perceptual_embedding(visual_emb, state)
            latent_goal = self.visual_goal(perceptual_emb[:, -1])
            # ------------Plan Proposal------------ #
            pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each
            sampled_plan = pp_dist.sample()  # sample from proposal net
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    def get_pp_plan_lang(
        self, curr_imgs: List[torch.Tensor], curr_state: torch.Tensor, goal_lang: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = [curr_img.unsqueeze(0) for curr_img in curr_imgs]
        curr_state = curr_state.unsqueeze(0)
        with torch.no_grad():
            visual_emb = self.visual_embedding(imgs)
            perceptual_emb = self.perceptual_embedding(visual_emb, curr_state)
            latent_goal = self.language_goal(goal_lang)
            # ------------Plan Proposal------------ #
            pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each
            sampled_plan = pp_dist.sample()  # sample from proposal net
        return sampled_plan, latent_goal

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        logger.info(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")