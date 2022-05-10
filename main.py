import torch
import torch.nn as nn
import h5py
import fire
import os
import numpy as np
import datetime
from pprint import pformat
from glob import glob
from tqdm import tqdm

import utils
import losses
import models
from dataloader import create_dataloader,\
    create_multi_instance_dataloader
import metrics

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint,\
    EarlyStopping, global_step_from_engine
from ignite.metrics import RunningAverage, Average

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:2'
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(device)

class Runner(object):

    def __init__(self, seed=0):
        super(Runner, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if 'cuda' in device:
            torch.cuda.manual_seed(seed)
        self.seed = seed
    
    @staticmethod
    def _forward(model, batch):
        inputs, targets, _, _ = batch
        inputs, targets = inputs.to(DEVICE, torch.float32),\
            targets.to(DEVICE, torch.float32)
        return inputs, model(inputs), targets

    def single(self, model_path=None, config='config/config.yaml', debug=False):
        config = utils.parse_config(config)
        outputdir = os.path.join(config['outputdir'], 
                "{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m')))
        os.makedirs(outputdir, exist_ok=True)
        logger = utils.genlogger(os.path.join(outputdir, 'logging.txt'))
        logger.info(f'Output directory is: {outputdir}')
        for k, v in config.items():
            logger.info(f'{k}: {v}')
        torch.save(config, os.path.join(outputdir, 'run_config.d'))

        train, dev, test = utils.split_dataset(
            config['input_h5'], debug=debug, seed=self.seed)
        TrainTransform, EvalTransform =\
            utils.TrainTransform, utils.EvalTransform

        TrainDataloader = create_dataloader(
            config['input_h5'], train, 
            transform=TrainTransform, 
            **config['dataloader_args'])
        DevDataloader = create_dataloader(
            config['input_h5'], dev,
            transform=EvalTransform, 
            **config['dataloader_args'])
        TestDataloader = create_dataloader(
            config['input_h5'], test,
            transform=EvalTransform, 
            **config['dataloader_args'])

        model = getattr(models, config['model'])(
            **config['model_args'])
        if model_path:
            file = glob(os.path.join(model_path, 'eval_best*.pt'))[0]
            params = torch.load(file, map_location='cpu')
            backbone_params = {k.replace('backbone.', ''): v for k, v in params.items() if 'backbone' in k}
            decoder_params = {k.replace('decoder.', ''): v for k, v in params.items() if 'decoder' in k}
            if decoder_params:
                model.load_param(backbone_params, decoder_params)
            else:
                model.load_param(backbone_params)
            logger.info(f'loading params from {model_path}')
        model.to(device)
        criterion = getattr(losses, config['criterion'])(**config['criterion_args'])
        optimizer = getattr(torch.optim, config['optimizer'])(
            model.parameters(), **config['optimizer_args'])
        scheduler = getattr(torch.optim.lr_scheduler, config['scheduler'])(
            optimizer, **config['scheduler_args'])


        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                inputs, outputs, targets = Runner._forward(model, batch)
                loss = criterion(outputs, targets, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.cpu().item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                inputs, outputs, targets = Runner._forward(model, batch)
                loss = criterion(outputs, targets, inputs)
            return loss.cpu().item()
        
        trainer, evaluator = Engine(_train), Engine(_inference)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        Average(output_transform=lambda x: x).attach(evaluator, 'Loss')
        ProgressBar(persist=False, ncols=75).attach(
            trainer, output_transform=lambda x: {'loss': x})
        ProgressBar(persist=False, ncols=75, desc='Evaluating').attach(
            evaluator, output_transform=None)

        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate(engine):
            logger.info(f'<==== Epoch {trainer.state.epoch} ====>')
            evaluator.run(DevDataloader)
            train_loss = engine.state.metrics['Loss']
            dev_loss = evaluator.state.metrics['Loss']
            logger.info('Training Loss: {:<5.2f}'.format(train_loss))
            logger.info('Validation Loss: {:<5.2f}'.format(dev_loss))
            scheduler.step(dev_loss)
        
        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def test(_):
            model.eval()
            f1_macro, f1_micro = metrics.evaluate(model, TestDataloader, DEVICE)
            logger.info('F1 macro: {:<5.2f}'.format(f1_macro))
            logger.info('F1 macro: {:<5.2f}'.format(f1_micro))

        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='eval_best',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, {'model': model})
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, BestModelCheckpoint, {'model': model})
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, EarlyStoppingHandler)

        trainer.run(TrainDataloader,
            max_epochs=config['n_epochs'])
        return outputdir
    
    def multiple(self, model_path=None, config='config/multiple.yaml', debug=False):
        config = utils.parse_config(config)
        outputdir = os.path.join(config['outputdir'], 
                "{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m')))
        os.makedirs(outputdir, exist_ok=True)
        logger = utils.genlogger(os.path.join(outputdir, 'logging.txt'))
        logger.info(f'Output directory is: {outputdir}')
        for k, v in config.items():
            logger.info(f'{k}: {v}')
        torch.save(config, os.path.join(outputdir, 'run_config.d'))

        train, dev, test = utils.split_dataset(
            config['input_h5'], debug=debug, seed=self.seed)
        TrainTransform, EvalTransform =\
            utils.TrainTransform, utils.EvalTransform

        TrainDataloader = create_multi_instance_dataloader(
            config['input_h5'], train,
            transform=TrainTransform, 
            seed=self.seed, shuffle=True,
            **config['dataloader_args'])
        DevDataloader = create_multi_instance_dataloader(
            config['input_h5'], dev,
            transform=EvalTransform, 
            seed=self.seed, shuffle=False,
            **config['dataloader_args'])
        TestDataloader = create_multi_instance_dataloader(
            config['input_h5'], test,
            transform=EvalTransform, 
            seed=self.seed, shuffle=False,
            **config['dataloader_args'])

        model = getattr(models, config['model'])(
            **config['model_args'])
        if model_path:
            file = glob(os.path.join(model_path, 'eval_best*.pt'))[0]
            params = torch.load(file, map_location='cpu')
            backbone_params = {k.replace('backbone.', ''): v for k, v in params.items() if 'backbone' in k}
            decoder_params = {k.replace('decoder.', ''): v for k, v in params.items() if 'decoder' in k}
            if decoder_params:
                model.load_param(backbone_params, decoder_params)
            else:
                model.load_param(backbone_params)
            logger.info(f'loading params from {model_path}')
        model.to(device)
        criterion = getattr(losses, config['criterion'])(**config['criterion_args'])
        optimizer = getattr(torch.optim, config['optimizer'])(
            model.parameters(), **config['optimizer_args'])
        scheduler = getattr(torch.optim.lr_scheduler, config['scheduler'])(
            optimizer, **config['scheduler_args'])


        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                inputs, outputs, targets = Runner._forward(model, batch)
                loss = criterion(outputs, targets, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.cpu().item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                inputs, outputs, targets = Runner._forward(model, batch)
                loss = criterion(outputs, targets, inputs)
            return loss.cpu().item()
        
        trainer, evaluator = Engine(_train), Engine(_inference)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        Average(output_transform=lambda x: x).attach(evaluator, 'Loss')
        ProgressBar(persist=False, ncols=75).attach(
            trainer, output_transform=lambda x: {'loss': x})
        ProgressBar(persist=False, ncols=75, desc='Evaluating').attach(
            evaluator, output_transform=None)

        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate(engine):
            logger.info(f'<==== Epoch {trainer.state.epoch} ====>')
            evaluator.run(DevDataloader)
            train_loss = engine.state.metrics['Loss']
            logger.info('Training Loss: {:<5.2f}'.format(train_loss))
            dev_loss = evaluator.state.metrics['Loss']
            logger.info('Validation Loss: {:<5.2f}'.format(dev_loss))
            scheduler.step(dev_loss)
        
        best_test = [0]
        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def test(_):
            model.eval()
            f1_macro, f1_micro = metrics.evaluate(model, TestDataloader, DEVICE)
            if (f1_macro + f1_micro) > best_test[0]:
                best_test[0] = f1_macro + f1_micro
                torch.save(model.state_dict(),
                    os.path.join(outputdir, f'best_test.pt'))
            logger.info('Test F1 macro: {:<5.2f}'.format(f1_macro))
            logger.info('Test F1 micro: {:<5.2f}'.format(f1_micro))

        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='eval_best',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, {'model': model})
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, BestModelCheckpoint, {'model': model})
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, EarlyStoppingHandler)

        trainer.run(TrainDataloader,
            epoch_length=config['n_iters'], max_epochs=config['n_epochs'])
        return outputdir

    def evaluate(self, path, multiple=False, debug=False):
        config = torch.load(os.path.join(path, 'run_config.d'))
        *_, test = utils.split_dataset(
            config['input_h5'], debug=debug, seed=self.seed)
        EvalTransform = utils.EvalTransform
        if multiple:
            TestDataloader = create_multi_instance_dataloader(
                config['input_h5'], test,
                transform=EvalTransform, 
                seed=self.seed, shuffle=False,
                **config['dataloader_args'])
        else:
            TestDataloader = create_dataloader(
                config['input_h5'], test,
                transform=EvalTransform, 
                **config['dataloader_args'])

        model = getattr(models, config['model'])(
            **config['model_args'])
        if path:
            file = glob(os.path.join(path, 'eval_best*.pt'))[0]
            params = torch.load(file, map_location='cpu')
            model.load_state_dict(params)
            # backbone_params = {k.replace('backbone.', ''): v for k, v in params.items() if 'backbone' in k}
            # decoder_params = {k.replace('decoder.', ''): v for k, v in params.items() if 'decoder' in k}
            # if decoder_params:
            #     model.load_param(backbone_params, decoder_params)
            # else:
            #     model.load_param(backbone_params)
        model.to(device)
        model.eval()
        if multiple:
            df, f1_macro, f1_micro = metrics.test_multiple(model, TestDataloader, DEVICE)
        else:
            df, f1_macro, f1_micro = metrics.test(model, TestDataloader, DEVICE)
        df.to_csv(os.path.join(path, 'test.csv'))
        print(df)
        print(f1_macro, f1_micro)

if __name__ == '__main__':
    fire.Fire(Runner)
