#-*- coding: utf-8 -*-
import os
from typing import Optional, Union, List, Dict
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import OPTIM_WRAPPERS

from package_utils.utils import make_dir
from configs.get_config import load_config
from models.builder import DETECTORS, MODELS
from datasets import *
from logs.logger import Logger, LOG_DIR
from models import *
from runner.engine.optim_wrappers import *


class Runner:
    """
    An unified interface for training/testing model
    args:
        cfg: A config dict or a path to config file
        workdir: path to store train/test logs and model checkpoints
    """
    def __init__(self, 
                 cfg,
                 workir=None):
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        if workir is None:
            workir = LOG_DIR
        
        logdir = workir+'/logdir'
        make_dir(logdir)
        logcheckpoints = workir+'/checkpoints'
        make_dir(logcheckpoints)

        self.logger = Logger(task=f'{self.cfg.TASK}', workdir=logdir)

    @classmethod
    def from_cfg(cls, cfg):
        """
        Utilize Config object from mmengine
        """
        if isinstance(cfg, str):
            cfg = Config.fromfile(cfg)
        
        return cls(cfg=cfg)

    def _init_model(self) -> nn.Module:
        """
        Initialize model for training/test
        """
        assert "MODEL" in self.cfg, "Model config must be set in config file!"   

        model_cfg = self.cfg.MODEL     
        model = DETECTORS.build(cfg=model_cfg)
        return model
    
    def init_model(self, optim_wrapper=None) -> nn.Module:
        """
        Loading model and initializing (pretrained) weights if applicable
        """
        self.model = self._init_model()
    
    def build_optim_wrapper(
        self, optim_wrapper: Union[Optimizer, OptimWrapper, Dict]
    ) -> Union[OptimWrapper, OptimWrapperDict]:
        """Build optimizer wrapper.

        If ``optim_wrapper`` is a config dict for only one optimizer,
        the keys must contain ``optimizer``, and ``type`` is optional.
        It will build a :obj:`OptimWrapper` by default.

        If ``optim_wrapper`` is a config dict for multiple optimizers, i.e.,
        it has multiple keys and each key is for an optimizer wrapper. The
        constructor must be specified since
        :obj:`DefaultOptimizerConstructor` cannot handle the building of
        training with multiple optimizers.

        If ``optim_wrapper`` is a dict of pre-built optimizer wrappers, i.e.,
        each value of ``optim_wrapper`` represents an ``OptimWrapper``
        instance. ``build_optim_wrapper`` will directly build the
        :obj:`OptimWrapperDict` instance from ``optim_wrapper``.

        Args:
            optim_wrapper (OptimWrapper or dict): An OptimWrapper object or a
                dict to build OptimWrapper objects. If ``optim_wrapper`` is an
                OptimWrapper, just return an ``OptimizeWrapper`` instance.

        Note:
            For single optimizer training, if `optim_wrapper` is a config
            dict, `type` is optional(defaults to :obj:`OptimWrapper`) and it
            must contain `optimizer` to build the corresponding optimizer.

        Examples:
            >>> # build an optimizer
            >>> optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=dict(
            ...     type='SGD', lr=0.01))
            >>> # optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> # is also valid.
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build optimizer without `type`
            >>> optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build multiple optimizers
            >>> optim_wrapper_cfg = dict(
            ...    generator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='SGD', lr=0.01)),
            ...    discriminator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='Adam', lr=0.001))
            ...    # need to customize a multiple optimizer constructor
            ...    constructor='CustomMultiOptimizerConstructor',
            ...)
            >>> optim_wrapper = runner.optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            name: generator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.1
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            name: discriminator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            'discriminator': Adam (
            Parameter Group 0
                dampening: 0
                lr: 0.02
                momentum: 0
                nesterov: False
                weight_decay: 0
            )

        Important:
            If you need to build multiple optimizers, you should implement a
            MultiOptimWrapperConstructor which gets parameters passed to
            corresponding optimizers and compose the ``OptimWrapperDict``.
            More details about how to customize OptimizerConstructor can be
            found at `optimizer-docs`_.

        Returns:
            OptimWrapper: Optimizer wrapper build from ``optimizer_cfg``.
        """  # noqa: E501
        if isinstance(optim_wrapper, OptimWrapper):
            return optim_wrapper
        if isinstance(optim_wrapper, (dict, ConfigDict, Config)):
            # optimizer must be defined for single optimizer training.
            optimizer = optim_wrapper.get('optimizer', None)

            # If optimizer is a built `Optimizer` instance, the optimizer
            # wrapper should be built by `OPTIM_WRAPPERS` registry.
            if isinstance(optimizer, Optimizer):
                optim_wrapper.setdefault('type', 'OptimWrapper')
                return OPTIM_WRAPPERS.build(optim_wrapper)  # type: ignore

            # If `optimizer` is not None or `constructor` is defined, it means,
            # optimizer wrapper will be built by optimizer wrapper
            # constructor. Therefore, `build_optim_wrapper` should be called.
            if optimizer is not None or 'constructor' in optim_wrapper:
                return build_optim_wrapper(self.model, optim_wrapper)
            else:
                # if `optimizer` is not defined, it should be the case of
                # training with multiple optimizers. If `constructor` is not
                # defined either, each value of `optim_wrapper` must be an
                # `OptimWrapper` instance since `DefaultOptimizerConstructor`
                # will not handle the case of training with multiple
                # optimizers. `build_optim_wrapper` will directly build the
                # `OptimWrapperDict` instance from `optim_wrapper.`
                optim_wrappers = OrderedDict()
                for name, optim in optim_wrapper.items():
                    if not isinstance(optim, OptimWrapper):
                        raise ValueError(
                            'each item mush be an optimizer object when '
                            '"type" and "constructor" are not in '
                            f'optimizer, but got {name}={optim}')
                    optim_wrappers[name] = optim
                return OptimWrapperDict(**optim_wrappers)
        else:
            raise TypeError('optimizer wrapper should be an OptimWrapper '
                            f'object or dict, but got {optim_wrapper}')
        
    def build_dataloaders(self):
        train_dataset = build_dataset(cfg=self.cfg.DATASET, 
                                      dataset=DATASETS, 
                                      default_args={'split': 'train', 'config': self.cfg.DATASET})
        train_loader = None
        val_loader = None
        test_loader = None

        return train_loader, val_loader, test_loader

    def train(self) -> nn.Module:
        """
        Main entry for training every model
        """
        assert isinstance(self.cfg, Config) or isinstance(self.cfg, dict), \
            "Config must be loaded before training!"
        
        # Loading model from registry
        self.init_model()
        n_param_model(self.model)
        self.logger.info(f'Loading {self.cfg.TASK} from model registry!')

        # Building datasets
        train_loader, val_loader, test_loader = self.build_dataloaders()

        # Building optimizer
        self.optim_wrapper = self.build_optim_wrapper(self.cfg.TRAIN.optim_wrapper)

        return self.model
