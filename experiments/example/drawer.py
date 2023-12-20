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
file_names = ['unc_rgb_db_0.15_n_40', 'unc_depth_db_0.2_n_40'] #['unc_depth_db_0.2_n_40' , 'unc_depth_db_0.0_n_40', 'unc_rgb_db_0.15_n_40',  'unc_rgb_db_0.0_n_40'] # 'unc_rgb_db_0.1_n_40','unc_rgb_db_0.2_n_40'


data_dicts = [] # to be left empty
for file_name in file_names:
    with open('./' + file_name + '.pkl', 'rb') as f:
        data = pickle.load(f) 
        data_dicts.append(data)



for img_id, img_data in data['heads'].items():
    plt.figure(figsize=(25, 10))

    for i, data in enumerate(data_dicts):
        # Display the original image
        
        import matplotlib.patches as patches

        img = img_data['orig'][0].cpu().numpy()  # Get the original image
        img = (img * 255).astype(np.uint8) 
        boxes = data['results'][img_id]  # Get the bounding boxes for the image

        parameters_to_channels = {
            'outs': 0, 
            'heatmap': 1, 
            'size_3d': 2, 
            'depth': 1,
            }


        n_drawings = len(parameters_to_channels) + 1
        ax = plt.subplot(len(data_dicts), n_drawings, 1+i*n_drawings)
        for box in boxes:
            # get the coordinates of the projection of 3d box - they are the last 16 elements of the box
            print("box", box)
            corners3d = np.array(box[-1]).reshape(8, 2)  # Reshape to 8 corners with 2 coordinates each

            # draw the corners on top of the image
            for k, corner in enumerate(corners3d):
                ax.scatter(corner[0], corner[1], c=[[k*31/255, k*31/255, k*31/255]], s=10)

            # draw the lines connecting the corners, making a cube
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical lines
            ]
            for line in lines:
                ax.plot(corners3d[line, 0], corners3d[line, 1], c='r', linewidth=1)

        # Display the image with the boxes
        plt.title(f'id={img_id}, {data["modality"]}, p={data["drop_prob"]}, bayes={data["bayes_n"]}')
        plt.imshow(img)

        # Loop over the parameters and display their uncertainty maps
        for j, (parameter, channel) in enumerate(parameters_to_channels.items()):
            if parameter == 'outs':
                parameter_data = data['outs'][img_id][channel].cpu()
            else:
                parameter_data = data['heads'][img_id][parameter][channel]
            plt.subplot(len(data_dicts), n_drawings, j+2+i*n_drawings)
            plt.title(f'Head: {parameter}, Channel: {channel}')
            plt.imshow(parameter_data.cpu().numpy())



    plt.tight_layout()
    plt.show()