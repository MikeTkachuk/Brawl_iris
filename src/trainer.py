import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict, namedtuple
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.agent import Agent
from src.collector import Collector
from src.envs import SingleProcessEnv, MultiProcessEnv
from src.episode import Episode
from src.dataset import EpisodesDataset
from src.make_reconstructions import make_reconstructions_from_batch
from src.models.actor_critic import ActorCritic
from src.models.world_model import WorldModel
from src.utils import configure_optimizer, EpisodeDirManager, set_seed
from src.aws import InstanceContext

import boto3


# TODO rewards adjust (currently -1 0 1)
class Trainer:
    def __init__(self, cfg: DictConfig, cloud_instance=False, env_actions=None) -> None:
        self.cloud_instance = cloud_instance
        if not self.cloud_instance:
            wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
                resume=True,
                **cfg.wandb
            )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)
        if self.cloud_instance:
            self.device = torch.device('cuda:0')

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        config_dir = Path('config')
        config_path = config_dir / 'trainer.yaml'
        config_dir.mkdir(exist_ok=False, parents=False)
        shutil.copy('.hydra/config.yaml', config_path)
        self.ckpt_dir.mkdir(exist_ok=True, parents=False)
        self.media_dir.mkdir(exist_ok=False, parents=False)
        self.episode_dir.mkdir(exist_ok=False, parents=False)
        self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train',
                                                  max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test',
                                                 max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination',
                                                             max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs,
                                   should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            self.train_dataset: EpisodesDataset = instantiate(cfg.datasets.train)
            if not self.cloud_instance:
                train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
                self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

        if self.cfg.evaluation.should:
            self.test_dataset: EpisodesDataset = instantiate(cfg.datasets.test)
            if not self.cloud_instance:
                test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
                self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        if not self.cloud_instance:
            env = train_env if self.cfg.training.should else test_env
        else:
            # if on cloud, env is only used for num_actions
            assert isinstance(env_actions, dict)
            env = namedtuple('Env', ['num_actions', 'num_continuous'])(**env_actions)

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions,
                                 act_continuous_size=env.num_continuous,
                                 config=instantiate(cfg.world_model))
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions,
                                   act_continuous_size=env.num_continuous)
        self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate,
                                                         cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(),
                                                       lr=cfg.training.learning_rate)

        if not self.cloud_instance:
            if cfg.initialization.storage_prefix is not None:
                os.system(f'aws s3 cp s3://{cfg.cloud.bucket_name}/'
                          f'{cfg.initialization.storage_prefix}/checkpoints/last.pt '
                          f'{self.ckpt_dir / "last.pt"}')
                cfg.initialization.path_to_checkpoint = self.ckpt_dir / "last.pt"
            if cfg.initialization.path_to_checkpoint is not None:
                self.agent.load(**cfg.initialization, device=self.device)

            if cfg.common.resume:
                os.system(f'aws s3 cp s3://{cfg.cloud.bucket_name}/{cfg.common.resume}/checkpoints '
                          f'{self.ckpt_dir} '
                          f'--recursive '
                          f'--quiet'
                          )
                self.load_checkpoint(load_dataset=False)
                self.train_dataset.num_seen_episodes = max(
                    [int(p.stem) for p in (self.ckpt_dir / 'dataset').glob('*')]
                ) + 1

    def benchmark(self):
        for i in range(20):
            self.train_collector.collect(self.agent, i, **self.cfg.collection.train.config)

    def run(self) -> None:

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []
            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)

                if self.cfg.training.on_cloud:
                    to_log += self.train_agent_cloud(epoch)
                else:
                    to_log += self.train_agent(epoch)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                self.test_dataset.clear()
                to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
                to_log += self.eval_agent(epoch)

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()

    def train_agent_cloud(self, epoch: int):
        """
        Uploads checkpoints to storage, launches optimization jobs on cloud,
        waits for the jobs to finish, downloads the results and resumes run.
        :param epoch:
        :return:
        """
        # defile idle clicker to keep the game active
        from controls import idle_click

        def _clicker(_stop_event: threading.Event):
            while True:
                idle_click()
                time.sleep(5)
                if _stop_event.is_set():
                    break

        stop_idle_clicking_key = threading.Event()
        idle_clicker = threading.Thread(target=_clicker, args=(stop_idle_clicking_key,))
        idle_clicker.start()

        # start initialization
        metrics = [{'epoch': epoch}]

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        # upload checkpoints and run jobs if any is ready to be optimized
        if any([
            epoch > cfg_tokenizer.start_after_epochs,
            epoch > cfg_world_model.start_after_epochs,
            epoch > cfg_actor_critic.start_after_epochs
        ]):

            # upload new and delete old episodes
            run_prefix = Path('_'.join([self.cfg.wandb.name, Path(os.getcwd()).parent.name, Path(os.getcwd()).name]))
            s3_client = boto3.client(
                's3'
            )

            to_upload, to_delete = [], []
            self.save_checkpoint(epoch, save_agent_only=False)
            local_dataset_filenames = set(file.name for file in (self.ckpt_dir / 'dataset').iterdir() if file.is_file())
            cloud_dataset_response = s3_client.list_objects_v2(
                Bucket=self.cfg.cloud.bucket_name,
                Prefix=str(run_prefix / 'checkpoints/dataset').replace('\\', '/')
            )
            cloud_dataset_filenames = set(
                Path(file_meta['Key']).name for file_meta in cloud_dataset_response.get('Contents', []))
            to_upload.extend([self.ckpt_dir / 'dataset' / file_name
                              for file_name in local_dataset_filenames.difference(cloud_dataset_filenames)])
            to_delete.extend(cloud_dataset_filenames.difference(local_dataset_filenames))

            # delete old episodes on cloud
            for file_name in to_delete:
                s3_client.delete_object(Bucket=self.cfg.cloud.bucket_name,
                                        Key=str(run_prefix / 'checkpoints/dataset' / file_name).replace('\\', '/'))

            # upload updated episodes and model checkpoints
            if s3_client.list_objects_v2(Bucket=self.cfg.cloud.bucket_name,
                                         Prefix=str(run_prefix / 'checkpoints').replace('\\', '/')
                                         )['KeyCount'] < 3:  # only the first time
                print('Trainer.train_agent_cloud: full checkpoint staged for upload')
                to_upload.extend([f for f in self.ckpt_dir.iterdir() if f.is_file()])
            else:
                to_upload.append(self.ckpt_dir / 'epoch.pt')  # always update epoch
            print(f'Trainer.train_agent_cloud: Started uploading {len(to_upload)} files')
            for file in tqdm(to_upload):
                name_on_bucket = str(run_prefix / 'checkpoints' / file.relative_to(self.ckpt_dir)).replace('\\', '/')
                s3_client.upload_file(str(file.absolute()), self.cfg.cloud.bucket_name, name_on_bucket)

            repo_root = Path(__file__).parents[1]  # ->Brawl_iris/src/trainer.py
            if s3_client.list_objects_v2(Bucket=self.cfg.cloud.bucket_name,
                                         Prefix=str(run_prefix / repo_root.name).replace('\\', '/')
                                         )['KeyCount'] < 3:  # only the first time
                print('Trainer.train_agent_cloud: code upload started')
                # upload code if needed
                name_on_bucket = str(run_prefix / f'{repo_root.name}').replace('\\', '/')
                os.system(f'aws s3 cp {repo_root} s3://{self.cfg.cloud.bucket_name}/{name_on_bucket} '
                          f'--exclude ".git/*" '
                          f'--exclude ".idea/*" '
                          f'--exclude "results/*" '
                          f'--exclude "assets/*" '
                          f'--recursive '
                          f'--quiet')
                print('Trainer.train_agent_cloud: code upload finished')

            # start job
            with InstanceContext(self.cfg.cloud.instance_id, region_name=self.cfg.cloud.region_name) as instance:
                instance.connect(self.cfg.cloud.key_file)
                time.sleep(5)  # prevents unfinished initializations
                instance.exec_command("rm -r Brawl_iris checkpoints")
                instance.exec_command(f"aws s3 cp \"s3://{self.cfg.cloud.bucket_name}/{run_prefix}\" ~ "
                                      f"--recursive "
                                      f"--quiet")
                env_actions = json.dumps({'num_actions': int(self.train_collector.env.num_actions),
                                          'num_continuous': int(self.train_collector.env.num_continuous)})
                # TODO (not important) save env_actions in config

                instance.exec_command(f"sh {repo_root.name}/aws_setup/run.sh {run_prefix}")

            # download checkpoint and metrics
            os.system(f'aws s3 cp s3://{self.cfg.cloud.bucket_name}/{run_prefix}/checkpoints {self.ckpt_dir} '
                      f'--exclude "dataset/*" '
                      f'--recursive')

            # load checkpoint locally
            self.load_checkpoint(load_dataset=False)
            with open(self.ckpt_dir / 'metrics.json') as metrics_file:
                metrics = json.load(metrics_file)
                metrics[0]['duration_gpu'] = instance.session_time / 3600
                print(
                    f'Trainer.train_agent_cloud: Received metrics: {metrics}')

        # terminate clicker
        stop_idle_clicking_key.set()
        idle_clicker.join()

        return metrics

    def train_agent(self, epoch: int):
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        w = self.cfg.training.sampling_weights

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer, sequence_length=1,
                                                     sample_from_start=True, sampling_weights=w, **cfg_tokenizer)
        self.agent.tokenizer.eval()

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.train_component(self.agent.world_model, self.optimizer_world_model,
                                                       sequence_length=self.cfg.common.sequence_length,
                                                       sample_from_start=True, sampling_weights=w,
                                                       tokenizer=self.agent.tokenizer, **cfg_world_model)
        self.agent.world_model.eval()

        if epoch > cfg_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic,
                                                        sequence_length=1 + self.cfg.training.actor_critic.burn_in,
                                                        sample_from_start=False, sampling_weights=w,
                                                        tokenizer=self.agent.tokenizer,
                                                        world_model=self.agent.world_model, **cfg_actor_critic)
        self.agent.actor_critic.eval()

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int,
                        batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float],
                        sequence_length: int, sampling_weights: Optional[Tuple[float]], sample_from_start: bool,
                        **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        for step in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout, mininterval=5):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch = self.train_dataset.sample_batch(batch_num_samples, sequence_length, sampling_weights,
                                                        sample_from_start)
                batch = self._to_device(batch)

                losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

                if step % 20 == 0:
                    print(f"Total Loss at step {step}: {loss_total_epoch * steps_per_epoch / (step + 1)}", flush=True)

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples,
                                                    sequence_length=1)

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples,
                                                      sequence_length=self.cfg.common.sequence_length,
                                                      tokenizer=self.agent.tokenizer)

        if epoch > cfg_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions:
            batch = self._to_device(
                self.test_dataset.sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch,
                                            tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss: Any) -> \
            Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes,
                                               sequence_length=1 + self.cfg.training.actor_critic.burn_in,
                                               sample_from_start=False)
        outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model,
                                                  horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        to_log = []
        for i, (o, a, ac, r, d) in enumerate(
                zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.actions_continuous.cpu(),
                    outputs.rewards.cpu(),
                    outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, ac, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(
                0) + i
            self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(),
                                                                  num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self, load_dataset=True, agent_only=False) -> None:
        assert self.ckpt_dir.is_dir()
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        if agent_only:
            print(f'Successfully loaded agent from {self.ckpt_dir.absolute()}.')
            return

        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        if load_dataset:
            self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(
            f'Successfully loaded model, optimizer '
            f'{f"and {len(self.train_dataset)} episodes" if load_dataset else ""} from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        wandb.finish()
