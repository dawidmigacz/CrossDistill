import pickle
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import cv2
import matplotlib.patches as patches
import math


os.environ["KMP_DUPLICATE_LIB_OK"] = "1"


# Load the data from the pickle file
file_names = ['unc_rgb_db_0.15_n_40'] #['unc_depth_db_0.2_n_40' , 'unc_depth_db_0.0_n_40', 'unc_rgb_db_0.15_n_40',  'unc_rgb_db_0.0_n_40'] # 'unc_rgb_db_0.1_n_40','unc_rgb_db_0.2_n_40'
data_dicts = []

for file_name in file_names:
    with open('./' + file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
        data_dicts.append(data)



for img_id, img_data in data['heads'].items():
    plt.figure(figsize=(15, 10))

    for i, data in enumerate(data_dicts):
        # Display the original image
        
        import matplotlib.patches as patches

        img = img_data['orig'][0].cpu().numpy()  # Get the original image
        # img = np.transpose(img, (1, 2, 0))  # Change the shape to (height, width, channels)
        img = (img * 255).astype(np.uint8)  # Change the data type to uint8
        boxes = data['results'][img_id]  # Get the bounding boxes for the image

        n_drawings = 5

        ax = plt.subplot(len(data_dicts), n_drawings, 1+i*(len(data_dicts)-1))
        for box in boxes:
            # x1, y1, x2, y2 = box[2:6]  # Get the coordinates of the box
            # get the coordinates of the projection of 3d box - they are the last 16 elements of the box
            corners3d = np.array(box[-1])
            # draw the corners on top of the image
            for k, corner in enumerate(corners3d):
                ax.scatter(corner[0], corner[1], c=[[k*31/255, k*31/255, k*31/255]], s=10)
            # draw the lines connecting the corners
            for i in range(4):
                ax.plot(corners3d[i:i+2, 0], corners3d[i:i+2, 1], c='r', linewidth=1)
            

            # Draw the bounding box on top of the image
            
            
            # score = box[-1]  # Get the confidence score
            # colour = (1 - (1 / (1 + math.exp(-score))), 1 / (1 + math.exp(-score)), 0)  # The higher the score, the greener the box
            # rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=colour, facecolor='none')
            # ax.add_patch(rect)


        # Display the image with the boxes
        
        plt.title(f'id={img_id}, {data["modality"]}, p={data["drop_prob"]}, bayes={data["bayes_n"]}')
        plt.imshow(img)

        # Display the first channel of outs
        outs = data['outs'][img_id][0].cpu()
        plt.subplot(len(data_dicts), n_drawings, 2+i*(len(data_dicts)-1))
        plt.title(f'Outs, Channel: 0')
        plt.imshow(outs)

        # Display the first channel of heads['heatmap']
        heatmap = data['heads'][img_id]['heatmap'][0]
        plt.subplot(len(data_dicts), n_drawings, 3+i*(len(data_dicts)-1))
        plt.title(f'Head: heatmap, Channel: 0')
        plt.imshow(heatmap.cpu().numpy())
        
        # Display the first channel of heads['size_3d']
        size_3d = data['heads'][img_id]['size_3d'][0]
        plt.subplot(len(data_dicts), n_drawings, 4+i*(len(data_dicts)-1))
        plt.title(f'Head: size_3d, Channel: 0')
        plt.imshow(size_3d.cpu().numpy())   

        # Display the first channel of heads['depth']
        depth = data['heads'][img_id]['depth'][0]
        plt.subplot(len(data_dicts), n_drawings, 5+i*(len(data_dicts)-1))
        plt.title(f'Head: depth, Channel: 0')
        plt.imshow(depth.cpu().numpy())



    plt.tight_layout()
    plt.show()