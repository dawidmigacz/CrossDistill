import argparse



import torch
import sys
import os
import yaml
import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.models.crossdistill_separate import CrossDistillSeparate
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint 
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

parser = argparse.ArgumentParser(description='Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('-u', '--uncertainty_only', action='store_true', default=False, help='uncertainty only')

args = parser.parse_args()

assert (os.path.exists(args.config))
cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
print(cfg)
set_random_seed(cfg.get('random_seed', 444))

log_path = ROOT_DIR + "/experiments/example/logs/"
if os.path.exists(log_path):
    pass
else:
    os.mkdir(log_path)
log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
logger = create_logger(log_path, log_file)

model = build_model(cfg['model'], flag='testing')

if os.path.exists(cfg['trainer']['pretrain_model']['rgb']):
    load_checkpoint(model=model.centernet_rgb,
                    optimizer=None,
                    filename=cfg['trainer']['pretrain_model']['rgb'],
                    map_location=device,
                    logger=logger)
else:
    logger.info("no rgb pretrained model")
    assert os.path.exists(cfg['trainer']['pretrain_model']['rgb'])

if os.path.exists(cfg['trainer']['pretrain_model']['depth']):
    load_checkpoint(model=model.centernet_depth,
                    optimizer=None,
                    filename=cfg['trainer']['pretrain_model']['depth'],
                    map_location=device,
                    logger=logger)
else:
    logger.info("no depth pretrained model")
    assert os.path.exists(cfg['pretrain_model']['depth'])

os.makedirs(cfg['trainer']['model_save_path'], exist_ok=True)
ckpt_name = os.path.join(cfg['trainer']['model_save_path'], 'MERGED')
save_checkpoint(get_checkpoint_state(model, None, 0), ckpt_name)

print(model)