from models.discriminator import UNetDiscriminatorAesrgan
from models.generator import Generator_RRDB
from loss.gan_loss import GANLoss
from loss.basic_loss import PerceptualLoss, L1Loss

# @title Trainer Code
# Trainer adapted from https://github.com/eugenesiow/super-image/blob/main/src/super_image/trainer.py
import os
import copy
import logging
from typing import Optional, Union, Dict, Callable
from collections import OrderedDict

from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler, Adam
from torch.nn.parallel import DataParallel, DistributedDataParallel

from super_image.modeling_utils import PreTrainedModel
from super_image.configuration_utils import PretrainedConfig
from super_image.file_utils import (
    WEIGHTS_NAME,
    WEIGHTS_NAME_SCALE,
    CONFIG_NAME
)
from super_image.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    set_seed
)
from super_image.training_args import TrainingArguments
from super_image.utils.metrics import AverageMeter, compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer is a simple class implementing the training and eval loop for PyTorch to train a super-image model.
    Args:
        model (:class:`~super_image.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
            .. note::
                :class:`~super_image.Trainer` is optimized to work with the :class:`~super_image.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the super_image models.
        args (:class:`~super_image.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~super_image.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset` or :obj:`torch.utils.data.dataset.IterableDataset`):
            The dataset to use for training.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation.
    """

    def __init__(
            self,
            net_g,
            net_d,
            opt_g,
            opt_d,
            num_epochs=100,
            batch_size=4,
            train_dataset: Dataset = None,
            eval_dataset: Optional[Dataset] = None,
    ):
        self.train_batch_size = batch_size
        self.num_train_epochs = num_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Seed must be set before instantiating the model when using model
        self.net_g = net_g.to(self.device)
        self.net_d = net_d.to(self.device)
        self.optimizer_d = opt_d
        self.optimizer_g = opt_g
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.best_epoch = 0
        self.best_metric = 0.0
        self.ema_decay = 0.999

        self.cri_pix = L1Loss().to(self.device)
        perceptual_weights = {
            'conv1_2': 0.1,
            'conv2_2': 0.1,
            'conv3_4': 1,
            'conv4_4': 1,
            'conv5_4': 1
        }
        self.cri_perceptual = PerceptualLoss(
            layer_weights=perceptual_weights
        ).to(self.device)
        self.cri_gan = GANLoss(gan_type='vanilla', loss_weight=1e-1).to(self.device)

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,

            **kwargs,
    ):
        """
        Main training entry point.
        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~super_image.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~super_image.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        self.net_g.train()
        self.net_d.train()
        epochs_trained = 0
        device = self.device
        num_train_epochs = self.num_train_epochs
        # learning_rate = args.learning_rate
        train_batch_size = self.train_batch_size
        train_dataset = self.train_dataset
        train_dataloader = self.get_train_dataloader()
        step_size = int(len(train_dataset) / train_batch_size * 200)

        # # Load potential model checkpoint
        # if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        #     resume_from_checkpoint = get_last_checkpoint(args.output_dir)
        #     if resume_from_checkpoint is None:
        #         raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
        #
        # if resume_from_checkpoint is not None:
        #     if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
        #         raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
        #
        #     logger.info(f"Loading model from {resume_from_checkpoint}).")
        #
        #     if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
        #         config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
        #
        #     state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
        #     # If the model is on the GPU, it still works!
        #     self._load_state_dict_in_model(state_dict)
        #
        #     # release memory
        #     del state_dict

        # if args.n_gpu > 1:
        #     self.model = nn.DataParallel(self.model)

        # optimizer = Adam(self.model.parameters(), lr=learning_rate)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.args.gamma)

        for epoch in range(epochs_trained, num_train_epochs):
            with tqdm(total=(len(train_dataset) - len(train_dataset) % train_batch_size)) as t:
                t.set_description(f'epoch: {epoch}/{num_train_epochs - 1}')

                for data in train_dataloader:
                    gt, out = data

                    l1_gt = gt
                    percep_gt = gt
                    gan_gt = gt

                    # optimize net_g
                    for p in self.net_d.parameters():
                        p.requires_grad = False

                    self.optimizer_g.zero_grad()
                    self.output = self.net_g(out)

                    l_g_total = 0
                    loss_dict = OrderedDict()

                    # pixel loss
                    if self.cri_pix:
                        l_g_pix = self.cri_pix(self.output, l1_gt)
                        l_g_total += l_g_pix
                        loss_dict['l_g_pix'] = l_g_pix
                    # perceptual loss
                    if self.cri_perceptual:
                        l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                        if l_g_percep is not None:
                            l_g_total += l_g_percep
                            loss_dict['l_g_percep'] = l_g_percep
                        if l_g_style is not None:
                            l_g_total += l_g_style
                            loss_dict['l_g_style'] = l_g_style
                    # gan loss
                    fake_g_preds = self.net_d(self.output)
                    loss_dict['l_g_gan'] = 0
                    for fake_g_pred in fake_g_preds:
                        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                        l_g_total += l_g_gan
                        loss_dict['l_g_gan'] += l_g_gan

                    l_g_total.backward()
                    self.optimizer_g.step()

                    # optimize net_d
                    for p in self.net_d.parameters():
                        p.requires_grad = True

                    self.optimizer_d.zero_grad()
                    # real
                    real_d_preds = self.net_d(gan_gt)
                    loss_dict['l_d_real'] = 0
                    loss_dict['out_d_real'] = 0
                    l_d_real_tot = 0
                    for real_d_pred in real_d_preds:
                        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                        l_d_real_tot += l_d_real
                        loss_dict['l_d_real'] += l_d_real
                        loss_dict['out_d_real'] += torch.mean(real_d_pred.detach())
                    l_d_real_tot.backward()
                    # fake
                    loss_dict['l_d_fake'] = 0
                    loss_dict['out_d_fake'] = 0
                    l_d_fake_tot = 0
                    fake_d_preds = self.net_d(self.output.detach().clone())  # clone for pt1.9
                    for fake_d_pred in fake_d_preds:
                        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                        l_d_fake_tot += l_d_fake
                        loss_dict['l_d_fake'] += l_d_fake
                        loss_dict['out_d_fake'] += torch.mean(fake_d_pred.detach())
                    l_d_fake_tot.backward()
                    self.optimizer_d.step()

                    # if self.ema_decay > 0:
                    #     self.model_ema(decay=self.ema_decay)

            # self.eval(epoch)

    def eval(self, epoch):

        scale = 4
        device = self.device
        eval_dataloader = self.get_eval_dataloader()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        self.net_g.eval()

        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = self.model(inputs)

            metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=scale)

            epoch_psnr.update(metrics['psnr'], len(inputs))
            epoch_ssim.update(metrics['ssim'], len(inputs))

        print(f'scale:{str(scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')

        if epoch_psnr.avg > self.best_metric:
            self.best_epoch = epoch
            self.best_metric = epoch_psnr.avg

            print(f'best epoch: {epoch}, psnr: {epoch_psnr.avg:.6f}, ssim: {epoch_ssim.avg:.6f}')
            self.save_model()

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        Will only save from the main process.
        """

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not isinstance(self.model, PreTrainedModel):
            # Setup scale
            scale = self.model.config.scale
            if scale is not None:
                weights_name = WEIGHTS_NAME_SCALE.format(scale=scale)
            else:
                weights_name = WEIGHTS_NAME

            weights = copy.deepcopy(self.model.state_dict())
            torch.save(weights, os.path.join(output_dir, weights_name))
        else:
            self.model.save_pretrained(output_dir)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    def get_eval_dataloader(self) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.
        """

        eval_dataset = self.eval_dataset
        if eval_dataset is None:
            eval_dataset = self.train_dataset

        return DataLoader(
            dataset=eval_dataset,
            batch_size=1,
        )
    # def get_bare_model(self, net):
    #     """Get bare model, especially under wrapping with
    #     DistributedDataParallel or DataParallel.
    #     """
    #     if isinstance(net, (DataParallel, DistributedDataParallel)):
    #         net = net.module
    #     return net
    # def model_ema(self, decay=0.999):
    #   net_g = self.get_bare_model(self.net_g)

    #   net_g_params = dict(net_g.named_parameters())
    #   net_g_ema_params = dict(self.net_g_ema.named_parameters())

    #   for k in net_g_ema_params.keys():
    #       net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)