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
file_names = ['unc_depth_db_0.2_n_40' , 'unc_depth_db_0.0_n_40', 'unc_rgb_db_0.2_n_40', 'unc_rgb_db_0.1_n_40', 'unc_rgb_db_0.0_n_40']
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

        # Create a subplot
        ax = plt.subplot(len(data_dicts), 3, 1+i*3)

        # Iterate over each bounding box
        for box in boxes:
            x1, y1, x2, y2 = box[2:6]  # Get the coordinates of the box
            score = box[-1]  # Get the confidence score

            # Map the confidence score to a color
            color = (1 - (1 / (1 + math.exp(-score))), 1 / (1 + math.exp(-score)), 0)  # The higher the score, the greener the box

            # Create a Rectangle patch
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        # Display the image with the boxes
        plt.title(f'Image ID: {img_id}, Modality: {data["modality"]}, Drop_prob: {data["drop_prob"]}, Bayes_n: {data["bayes_n"]}')
        plt.imshow(img)

        # Display the first channel of heads['heatmap']
        heatmap = data['heads'][img_id]['heatmap'][0]
        plt.subplot(len(data_dicts), 3, 3+i*3)
        plt.title(f'Head: heatmap, Channel: 0')
        plt.imshow(heatmap.cpu().numpy())
        
        # Display the first channel of outs
        outs = data['outs'][img_id][0].cpu()
        plt.subplot(len(data_dicts), 3, 2+i*3)
        plt.title(f'Outs, Channel: 0')
        plt.imshow(outs)

    plt.tight_layout()
    plt.show()