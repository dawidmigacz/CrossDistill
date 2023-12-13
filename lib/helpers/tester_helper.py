import os
import tqdm
import pickle
import torch

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
#import plt
import matplotlib.pyplot as plt


class Tester(object):
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


    def test(self):
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
            if self.bayes_n is None:
                self.evaluate()

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
                if self.bayes_n is None:
                    self.evaluate()



    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        heads={}
        outs = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, _, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            #inputs = inputs.to(self.device)
            
            if self.bayes_n is None:
                _, outputs, _ = self.model(inputs)
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
                                        threshold=self.cfg.get('threshold', 0.2))
                results.update(dets)

            else:       #self_bayes not none; ASSUME BATCH = 1
                heads_names = ['heatmap', 'offset_2d', 'size_2d' , 'depth', 'offset_3d', 'size_3d', 'heading']
                
                heads[info['img_id'][0].item()] =  {h: [] for h in ['orig'] + heads_names} 
                inp = torch.permute(inputs['rgb'], (0, 2, 3, 1))
                inp = (inp - inp.min()) / (inp.max() - inp.min())
                heads[info['img_id'][0].item()]['orig'] = inp
                outputs = 0
                for i in range(self.bayes_n):
                    _, outputs, _ = self.model(inputs)
                    # if i > 0:
                    #     heatmap1 = outputs['heatmap'][0][0].detach().cpu().numpy()
                    #     plt.imshow(heatmap-heatmap1, cmap='hot', interpolation='nearest')
                    #     plt.show()
                    # heatmap = outputs['heatmap'][0][0].detach().cpu().numpy()
                    
                    for head in outputs:
                        heads[info['img_id'][0].item()][head].append(outputs[head])

                outs[info['img_id'][0].item()] = outputs['heatmap'][0].clone().detach()
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
                                        threshold=self.cfg.get('threshold', 0.2))
                results.update(dets)
                # print(results[ info['img_id'][0].item() ])







            progress_bar.update()
        
        filename = './unc_' + str(self.model.modality) + '_db_' + str(self.model.drop_prob) + '_n_' + str(self.bayes_n) + '.pkl'


        if self.bayes_n is not None:
            # with open('unc_no_var.pkl', 'wb') as handle:
            #     pickle.dump(heads, handle, protocol=pickle.HIGHEST_PROTOCOL)

            for i in heads:
                for head in heads[i]:
                    if head != 'orig':
                        heads[i][head] = torch.cat(heads[i][head])
                        heads[i][head] = torch.var(heads[i][head], dim=0)
                        print(head, heads[i][head].max() , heads[i][head].min())
                        heads[i][head] = (heads[i][head] - heads[i][head].min()) / (heads[i][head].max() - heads[i][head].min())


        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        if self.bayes_n is None:
            self.save_results(results)
        else:
            with open(filename, 'wb') as handle:
                pickle.dump( {'heads': heads, 'outs': outs, 'modality': self.model.modality, 'drop_prob': self.model.drop_prob, 'bayes_n': self.bayes_n, 'results': results}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        self.logger.info('==> Results Saved !')



    def save_results(self, results, output_dir='./rgb_outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()



    def evaluate(self):
        _ = self.dataloader.dataset.eval(results_dir='./rgb_outputs/data', logger=self.logger)




