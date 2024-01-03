import os
import tqdm
import pickle
import torch

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
#import plt
import matplotlib.pyplot as plt
import numpy as np


class Uncertainter(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False):
        self.cfg = cfg
        self.model = model
        self.bayes_n = self.cfg.get('bayes_n', None)
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = './outputs'
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.eval = eval


    def uncertainty(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single':
            assert os.path.exists(self.cfg['checkpoint'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=self.cfg['checkpoint'],
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()


        # test all checkpoints in the given dir
        if self.cfg['mode'] == 'all':
            checkpoints_list = []
            for _, _, files in os.walk(self.cfg['checkpoints_dir']):
                checkpoints_list = [os.path.join(self.cfg['checkpoints_dir'], f) for f in files if f.endswith(".pth")]
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()



    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, _, info) in enumerate(self.dataloader):
            results = {}
            heads={}
            outs = {}
            img_id = info['img_id'][0].item()
            # load evaluation data and move data to GPU.
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            #inputs = inputs.to(self.device)
            
            #self_bayes not none; ASSUME BATCH = 1
            heads_names = ['heatmap', 'offset_2d', 'size_2d' , 'depth', 'offset_3d', 'size_3d', 'heading']
            
            heads[img_id] =  {h: [] for h in ['orig'] + heads_names} 
            inp = torch.permute(inputs['rgb'], (0, 2, 3, 1))
            inp = (inp - inp.min()) / (inp.max() - inp.min())
            heads[img_id]['orig'] = inp
            outputs = 0
            for i in range(self.bayes_n):
                _, outputs, _ = self.model(inputs)


                for head in outputs:
                    heads[img_id][head].append(outputs[head].clone().detach())

            outs[img_id] = outputs['heatmap'][0].clone().detach()
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
            dets = dets.detach().cpu().numpy()


            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(dets=dets,
                                    info=info,
                                    calibs=calibs,
                                    cls_mean_size=cls_mean_size,
                                    threshold=self.cfg.get('threshold', 0.2),
                                    d3box=True)
            results.update(dets)        
            for i in heads:
                for head in heads[i]:
                    if head != 'orig':
                        heads[i][head] = torch.cat(heads[i][head])
                        heads[i][head] = torch.var(heads[i][head], dim=0)
                        # heads[i][head] = (heads[i][head] - heads[i][head].min()) / (heads[i][head].max() - heads[i][head].min())

            filename = '../../data/KITTI/object/training/unc_' + str(self.model.modality) +  '/' + str(img_id).zfill(6) + '.pkl'
            # self.logger.info('==> Saving ...')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as handle:
                pickle.dump( {'heads': heads, 'outs': outs, 'modality': self.model.modality, 'drop_prob': self.model.drop_prob, 'bayes_n': self.bayes_n, 'results': results}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # print(filename, ' saved')
            # self.logger.info('==> Results Saved !', filename)

            progress_bar.update(1)

        progress_bar.close()

        # save results
        




