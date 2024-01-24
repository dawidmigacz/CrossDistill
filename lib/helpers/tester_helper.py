import os
import tqdm
import pickle
import torch
import random

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
#import plt
import matplotlib.pyplot as plt
import numpy as np


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = './outputs'
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.eval = eval
        self.modality = cfg.get('modality', None)


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
                self.evaluate()



    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        selected_depth = []
        selected_rgb = []
        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        switches = 0
        num_images = 0
        self.num_img = len(self.dataloader)
        os.makedirs('depth_images', exist_ok=True)
        os.makedirs('rgb_images', exist_ok=True)
        for batch_idx, (inputs, targets, info) in enumerate(self.dataloader):
            img_id = info['img_id'][0].item()
            
            if(self.cfg['model_type']=="distill_separate"):

                unc_rgb = targets['unc_rgb']
                unc_depth = targets['unc_depth']
                total_sum_depth = torch.zeros((96, 320))
                total_sum_rgb = torch.zeros((96, 320))
                parameters = ['heatmap', 'size_3d', 'depth', 'offset_2d', 'size_2d', 'offset_3d', 'heading']

                for parameter in parameters:
                    total_sum_depth += unc_depth['heads'][img_id][parameter].sum(dim=0).sum(dim=0).cpu()
                    total_sum_rgb += unc_rgb['heads'][img_id][parameter].sum(dim=0).sum(dim=0).cpu()


                from torchvision.transforms import GaussianBlur
                from torchvision.transforms.functional import to_pil_image, to_tensor

                # Define the GaussianBlur transform
                transform = GaussianBlur(3)

                # Convert the tensors to PIL images, apply the transform, and convert back to tensors
                total_sum_depth = to_tensor(transform(to_pil_image(total_sum_depth)))
                total_sum_rgb = to_tensor(transform(to_pil_image(total_sum_rgb)))


 
            # load evaluation data and move data to GPU.
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            #inputs = inputs.to(self.device)

            if(self.cfg['model_type']=="distill_separate"):
                _, outputs, _ = self.model.centernet_rgb(inputs)
            else:
                _, outputs, _ = self.model(inputs)
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            r = {}
            dets = decode_detections(dets=dets,
                                    info=info,
                                    calibs=calibs,
                                    cls_mean_size=cls_mean_size,
                                    threshold=self.cfg.get('threshold', 0.2))
            r.update(dets)


            # Initialise variables to store the total uncertainties and the number of boxes
            total_uncertainty_depth = 0
            total_uncertainty_rgb = 0
            num_boxes = 0

            # print(r[img_id])
            if(self.cfg['model_type']=="distill_separate"):
            # Iterate over each bounding box
                for box in r[img_id]:
                    x1, y1, x2, y2 = box[2:6]  # Get the coordinates of the box
                    # centre 
                    x = int((x1 + x2)/2)
                    y = int((y1 + y2)/2)

                    # get the uncertainty
                    # print('shape of total_sum_depth: ', total_sum_depth.size())

                    # make sure y//4 and x//4 are in bounds
                    yy = y//4
                    xx = x//4
                    if yy>95:
                        yy = 95
                    if xx>319:
                        xx = 319
                    if yy<0:
                        yy=0
                    if xx<0:
                        xx = 0
                    
                    try:
                        uncertainty_depth = total_sum_depth[0, yy, xx]
                        uncertainty_rgb = total_sum_rgb[0, yy, xx]
                    except:
                        print(xx, yy)
                        raise KeyError
                    # Add the uncertainties to the total uncertainties
                    total_uncertainty_depth += uncertainty_depth
                    total_uncertainty_rgb += uncertainty_rgb

                    # Increment the number of boxes
                    num_boxes += 1

                # Calculate the average uncertainties
                if False: #num_boxes > 0:
                    average_uncertainty_depth = total_uncertainty_depth / num_boxes
                    average_uncertainty_rgb = total_uncertainty_rgb / num_boxes
                else:
                    average_uncertainty_depth = total_sum_depth.mean().item()
                    average_uncertainty_rgb = total_sum_rgb.mean().item()
                decision_switch = False
                if average_uncertainty_depth < average_uncertainty_rgb + self.cfg['uncertainty_threshold']:
                # if random.random() > self.cfg['uncertainty_threshold']:
                    switches += 1
                    decision_switch = True
                # print(f'{img_id}, {decision_switch}, {average_uncertainty_depth}, {average_uncertainty_rgb}, {average_uncertainty_depth-average_uncertainty_rgb}\n')
                if False:
                    # Tworzenie folderu, jeśli jeszcze nie istnieje
                    os.makedirs('uncertainties', exist_ok=True)

                    # Zapisywanie identyfikatora obrazu i niepewności do pliku
                    with open(os.path.join('uncertainties', 'uncertainties2.txt'), 'a') as f:
                        f.write(f'{img_id}, {average_uncertainty_depth}, {average_uncertainty_rgb}, {average_uncertainty_depth-average_uncertainty_rgb}\n')

                num_images += 1
  
                if(decision_switch):
                    selected_depth.append(img_id)
                    _, outputs, _ = self.model.centernet_depth(inputs)
                    dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
                    dets = dets.detach().cpu().numpy()
                    dets = decode_detections(dets=dets,
                                    info=info,
                                    calibs=calibs,
                                    cls_mean_size=cls_mean_size,
                                    threshold=self.cfg.get('threshold', 0.2))
                else:
                    selected_rgb.append(img_id)
            results.update(dets)

            progress_bar.update()
        progress_bar.close()
        if self.cfg['model_type']=="distill_separate":
            percentage_switches = (switches / num_images) * 100
            print('Percentage of images where depth uncertainty is smaller than rgb uncertainty: ', percentage_switches)
            self.percentage_switches = percentage_switches
            # Zapisywanie wybranych obrazów głębi do pliku
            with open(os.path.join('2depth_images', f'depth{percentage_switches}.txt'), 'w') as f:
                f.writelines(f'{img_id}\n' for img_id in selected_depth)

            # Zapisywanie wybranych obrazów RGB do pliku
            with open(os.path.join('2rgb_images', f'rgb{percentage_switches}.txt'), 'w') as f:
                f.writelines(f'{img_id}\n' for img_id in selected_rgb)

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)
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
        res = self.dataloader.dataset.eval(results_dir='./rgb_outputs/data', logger=self.logger)
        if self.cfg['model_type']=="distill_separate":
            print(self.percentage_switches, res['Car_3d_easy_R40'], res['Car_3d_moderate_R40'], res['Car_3d_hard_R40'])
            with open("wykres_dane.txt", "a") as myfile:
                myfile.write(str(self.percentage_switches) + " " + str(res['Car_3d_easy_R40']) + " " + str(res['Car_3d_moderate_R40']) + " " + str(res['Car_3d_hard_R40']) + " " + str(self.cfg["uncertainty_threshold"]) + "\n")
        if self.cfg['model_type']=="centernet3d":
            print(self.num_img, res['Car_3d_easy_R40'], res['Car_3d_moderate_R40'], res['Car_3d_hard_R40'])
            with open("tabelka_nieswitch.txt", "a") as myfile:
                myfile.write(self.modality + str(self.num_img) + " " + str(res['Car_3d_easy_R40']) + " " + str(res['Car_3d_moderate_R40']) + " " + str(res['Car_3d_hard_R40']) + " " + str(self.cfg["uncertainty_threshold"]) + "\n")



