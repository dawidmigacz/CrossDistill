import os
import sys
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"


import yaml
import argparse
import datetime
import os

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.uncertainty_helper import Uncertainter

import numpy as np
import torch

parser = argparse.ArgumentParser(description='Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('-u', '--uncertainty_only', action='store_true', default=False, help='uncertainty only')

args = parser.parse_args()



def main():
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    from subprocess import call
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print("OS: ", sys.platform)
    print("Python: ", sys.version)
    print("PyTorch: ", torch.__version__)
    print("Numpy: ", np.__version__)
    np.random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    print(cfg)


    log_path = ROOT_DIR + "/experiments/example/logs/"
    if os.path.exists(log_path):
        pass
    else:
        os.mkdir(log_path)
    log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = create_logger(log_path, log_file)


    # build dataloader
    train_loader, test_loader  = build_dataloader(cfg['dataset'])

    wandb.init(
    project=cfg['wandb']['project'],
    notes=cfg['wandb']['notes'],
    tags=cfg['wandb']['tags'],
    config = cfg
    )

    if args.uncertainty_only:
        if cfg['tester'].get('bayes_n', None) is None:
            raise ValueError('bayes_n should be set in tester when uncertainty_only is True')
        model = build_model(cfg['model'], 'testing')
        # print(model)
        logger.info('###################  Uncertainty Only  ##################')
        uncertainter = Uncertainter(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        uncertainter.uncertainty()
        return

    if args.evaluate_only:
        # if cfg['tester'].get('bayes_n', None) is not None:
        #     raise ValueError('bayes_n should not be set in tester when evaluate_only is True')
        model = build_model(cfg['model'], 'testing')
        # print(model)
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()
        return




    # build model&&build optimizer
    if cfg['model']['type']=='centernet3d' or cfg['model']['type']=='distill':
        model = build_model(cfg['model'],'training')
        print(model.modality)
        print(model)

        optimizer = build_optimizer(cfg['optimizer'], model)
        lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    else:
        raise NotImplementedError("%s model is not supported" % cfg['model']['type'])


    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      model_type=cfg['model']['type'],
                      root_path=ROOT_DIR)
    trainer.train()

    logger.info('###################  Evaluation  ##################' )
    test_model_list = build_model(cfg['model'],'testing')
    cfg['tester']['uncertainty_threshold'] = np.random.uniform(-1.0, 1.0)
    tester = Tester(cfg=cfg['tester'],
                    model=test_model_list,
                    dataloader=test_loader,
                    logger=logger)
    tester.test()


if __name__ == '__main__':
    main()