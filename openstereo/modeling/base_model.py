"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `openstereo/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_val(model)
"""
import os
import os.path as osp
from abc import ABCMeta
from abc import abstractmethod
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.dataset import DataSet
from evaluation import evaluator as eval_functions
from utils import NoOp
from utils import Odict, mkdir, ddp_all_gather
from utils import get_msg_mgr
from utils import get_valid_args, is_list, is_dict, ts2np, get_attr_from
from . import backbone
from . import cost_processor
from . import disp_processor
from .loss_aggregator import LossAggregator

__all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """

    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self, *args, **kwargs):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(self, *args, **kwargs):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_val(self, *args, **kwargs):
        """Run a whole test schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(self, *args, **kwargs):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, scope='train'):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.epoch = 0
        self.scope = scope
        is_train = scope == 'train'

        self.engine_cfg = cfgs['trainer_cfg'] if is_train else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if is_train and self.engine_cfg['enable_float16']:
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()

        self.msg_mgr.log_info(cfgs['data_cfg'])

        if scope == 'train':
            self.train_loader = self.get_loader(cfgs['data_cfg'], 'train')
            if self.engine_cfg['with_test']:
                self.val_loader = self.get_loader(cfgs['data_cfg'], 'val')
        elif scope == 'val':
            self.val_loader = self.get_loader(cfgs['data_cfg'], 'val')
        else:
            self.test_loader = self.get_loader(cfgs['data_cfg'], 'test')

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        # self.to(device=torch.device("cuda", self.device_rank))
        self.to(self.device)

        if is_train:
            self.loss_aggregator = self.get_loss_func(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])

        self.train(is_train)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def get_loss_func(self, loss_cfg):
        """Get the loss function."""
        return LossAggregator(loss_cfg)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbone], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg) for cfg in backbone_cfg])
            return Backbone
        raise ValueError("Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def get_cost_processor(self, cost_processor_cfg):
        """Get the backbone of the model."""
        if is_dict(cost_processor_cfg):
            CostProcessor = get_attr_from([cost_processor], cost_processor_cfg['type'])
            valid_args = get_valid_args(CostProcessor, cost_processor_cfg, ['type'])
            return CostProcessor(**valid_args)
        if is_list(cost_processor_cfg):
            CostProcessor = nn.ModuleList([self.get_cost_processor(cfg) for cfg in cost_processor_cfg])
            return CostProcessor
        raise ValueError("Error type for -Cost-Processor-Cfg-, supported: (A list of) dict.")

    def get_disp_processor(self, disp_processor_cfg):
        """Get the backbone of the model."""
        if is_dict(disp_processor_cfg):
            DispProcessor = get_attr_from([disp_processor], disp_processor_cfg['type'])
            valid_args = get_valid_args(DispProcessor, disp_processor_cfg, ['type'])
            return DispProcessor(**valid_args)
        if is_list(disp_processor_cfg):
            DispProcessor = nn.ModuleList([self.get_cost_processor(cfg) for cfg in disp_processor_cfg])
            return DispProcessor
        raise ValueError("Error type for -Disp-Processor-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['backbone_cfg'])
            self.Backbone = self.get_backbone(cfg)
        if 'cost_processor_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['cost_processor_cfg'])
            self.CostProcessor = self.get_cost_processor(cfg)
        if 'disp_processor_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['disp_processor_cfg'])
            self.DispProcessor = self.get_disp_processor(cfg)

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)
        return disp_out

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def get_loader(self, data_cfg, scope):
        is_train = scope == 'train'
        dataset = DataSet(data_cfg, scope)
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if is_train else self.cfgs['evaluator_cfg']['sampler']
        sampler = DistributedSampler(
            dataset,
            shuffle=sampler_cfg['batch_shuffle'] if is_train else False,
            drop_last=False,
        )
        # Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        # vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
        #     'sample_type', 'type'])
        # print(vaild_args)
        # sampler = Sampler(dataset)
        # print(sampler)
        # collate_fn = CollateFn(dataset.label_set, sampler_cfg)

        loader = DataLoader(
            dataset=dataset,
            batch_size=sampler_cfg['batch_size'],
            sampler=sampler,
            num_workers=data_cfg['num_workers'],
            drop_last=False,
            # pin_memory=True,
            # batch_sampler=sampler,
            # collate_fn=collate_fn,
        )
        return loader

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(params=filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from([optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration=None, epoch=None):
        if dist.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': self.iteration if iteration is None else iteration,
                'epoch': self.epoch if epoch is None else epoch,
            }
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, self.iteration)))
            self.msg_mgr.log_info('Save checkpoints in checkpoints/{}-{:0>5}.pt'.format(save_name, self.iteration))
        torch.distributed.barrier()

    def _load_ckpt(self, save_name):
        map_location = {"cuda:0": f"cuda:{os.environ['LOCAL_RANK']}"}
        checkpoint = torch.load(
            save_name,
            map_location=map_location
        )
        model_state_dict = checkpoint['model']

        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']
        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def inputs_pretreament(self, inputs):
        """Reorganize input data for different models

        Args:
            inputs: the input data.
        Returns:
            dict: training data including ref_img, tgt_img, disp image,
                  and other meta data.
        """
        # asure the disp_gt has the shape of [B, H, W]
        disp_gt = inputs['disp']
        if len(disp_gt.shape) == 4:
            disp_gt = disp_gt.squeeze(1)

        # compute the mask of valid disp_gt
        max_disp = self.cfgs['model_cfg']['base_config']['max_disp']
        mask = (disp_gt < max_disp) & (disp_gt > 0)

        return {
            'ref_img': inputs['left'].to(self.rank),
            'tgt_img': inputs['right'].to(self.rank),
            'disp_gt': disp_gt.to(self.rank),
            'mask': mask.to(self.rank),
        }

    def train_step(self, loss_sum) -> bool:
        """
        Conduct:
            loss_sum.backward()
            self.optimizer.step()
            self.scheduler.step()
        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug(
                    "Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                        scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    @staticmethod
    def inference(model, loader):
        """
        Inference all the test data.
        Args:
            model: the model to be tested.
            loader: data loader.
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(loader)
        if model.device_rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in loader:
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                output = model(ipts)
                inference_disp = output['inference_disp']
                for k, v in inference_disp.items():
                    inference_disp[k] = ddp_all_gather(v, requires_grad=False)
                del output
            inference_disp.update(ipts)
            for k, v in inference_disp.items():
                inference_disp[k] = ts2np(v)
            info_dict.append(inference_disp)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        # the final output is a dict {'disp_est': np.array}
        return info_dict

    def run_val(self, *args, **kwargs):
        """Accept the instance object(model) here, and then run the test loop."""
        self.eval()
        dataloader = self.val_loader
        eval_func = self.cfgs['evaluator_cfg']["eval_func"]
        eval_func = getattr(eval_functions, eval_func)
        valid_args = get_valid_args(eval_func, self.cfgs["evaluator_cfg"], ['metric'])
        eval_func = partial(eval_func, **valid_args)

        infoList = []

        total_size = len(dataloader) * dataloader.batch_sampler.batch_size * self.world_size
        rest_size = total_size
        batch_size = dataloader.batch_sampler.batch_size
        show_progress_bar = self.cfgs['evaluator_cfg']['show_progress_bar']
        if show_progress_bar and self.rank == 0:
            pbar = tqdm(total=total_size, desc='Evaluating')
        else:
            pbar = NoOp()
        self.msg_mgr.log_info(f"Total size: {total_size} | Total batch: {len(dataloader)}")
        with torch.no_grad():
            for inputs in dataloader:
                ipts = self.inputs_pretreament(inputs)
                with autocast(enabled=self.engine_cfg['enable_float16']):
                    # print(ipts['ref_img'].device)
                    output = self.forward(ipts)
                    inference_disp, visual_summary = output['inference_disp'], output['visual_summary']
                    inference_disp.update(ipts)
                    self.msg_mgr.write_to_tensorboard(visual_summary)
                    for k, v in inference_disp.items():
                        try:
                            inference_disp[k] = ts2np(v)
                        except:
                            print(k, "is not tensor")
                    info = eval_func(inference_disp)
                    infoList.append(info)
                rest_size -= batch_size
                update_size = batch_size if rest_size >= 0 else total_size % batch_size
                pbar.update(update_size)
        pbar.close()

        world_size = dist.get_world_size()

        # gather all the info from different processes
        output = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(output, infoList)

        # calculate the mean of the matric
        if self.rank == 0:
            all_info = []
            for rank in range(world_size):
                all_info.extend(output[rank])
            res_dict_keys = all_info[0].keys()
            res_dict = {k: [] for k in res_dict_keys}
            for info in all_info:
                for k, v in info.items():
                    res_dict[k].append(v)
            for k, v in res_dict.items():
                res_dict[k] = np.nanmean(v)
            visual_summary.update(res_dict)
            return res_dict

    def run_test(self, *args, **kwargs):
        try:
            self.resume_ckpt(self.cfgs['evaluator_cfg']['checkpoint'])
        except Exception as e:
            self.msg_mgr.log_warning("Failed to resume the checkpoint, got {}".format(e))
        with torch.no_grad():
            loader = self.test_loader
            info_dict = self.inference(self, loader)
        return info_dict

    def run_train(self, *args, **kwargs):
        """Accept the instance object(model) here, and then run the train loop."""
        self.train()
        while True:
            self.epoch += 1
            self.train_loader.sampler.set_epoch(self.epoch)  # for distributed training shuffle
            self.msg_mgr.log_info(f"Epoch {self.epoch} starts.")
            torch.cuda.empty_cache()
            for inputs in self.train_loader:
                ipts = self.inputs_pretreament(inputs)
                with autocast(enabled=self.engine_cfg['enable_float16']):
                    output = self.forward(ipts)
                    training_disp, visual_summary = output['training_disp'], output['visual_summary']
                    del output
                loss_sum, loss_info = self.loss_aggregator(training_disp)
                ok = self.train_step(loss_sum)
                if not ok:
                    self.msg_mgr.log_warning("Loss is NaN or Inf. Skip this iter.")
                    continue
                visual_summary.update(loss_info)
                current_lr = self.optimizer.param_groups[0]['lr']
                visual_summary['scalar/learning_rate'] = current_lr
                loss_info['scalar/learning_rate'] = current_lr

                self.msg_mgr.train_step(loss_info, visual_summary)

                if self.iteration % self.engine_cfg['save_iter'] == 0:
                    # save the checkpoint
                    self.save_ckpt()
                    # run test if with_test is true
                    if self.engine_cfg['with_test']:
                        self.msg_mgr.log_info("Running test...")
                        self.eval()
                        result_dict = self.run_val()
                        torch.cuda.empty_cache()
                        self.msg_mgr.log_info(result_dict)
                        self.msg_mgr.write_to_tensorboard(result_dict)
                        self.msg_mgr.reset_time()
                        self.train()
                        if self.cfgs['trainer_cfg']['fix_BN']:
                            self.fix_BN()

                if self.engine_cfg['total_epoch'] is None and self.iteration >= self.engine_cfg['total_iter']:
                    self.msg_mgr.log_info('Training finished! Reached the maximum iteration.')
                    self.save_ckpt()
                    return

            if self.epoch >= self.engine_cfg['total_epoch']:
                self.save_ckpt()
                self.msg_mgr.log_info('Training finished! Reached the maximum epoch.')
                return
