import pickle
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"


# Load the data from the pickle file
file_names = ['unc_depth_db_0.2_n_40', 'unc_rgb_db_0.2_n_40', 'unc_rgb_db_0.1_n_40', 'unc_rgb_db_0.0_n_40']
data_dicts = []

for file_name in file_names:
    with open('./' + file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
        data_dicts.append(data)

# Iterate over the data and display the images

# print(data)



# structure of data: {'heads': heads, 'outs': outs, 'modality': self.model.modality, 'drop_prob': self.model.drop_prob, 'bayes_n': self.bayes_n}

# for img_id, img_data in data.items():
#     plt.figure(figsize=(10, 10))
#     for i, (head, img) in enumerate(img_data.items()):
#         # Get the first channel
#         img_channel = img[0]
#         plt.subplot(4, 2, i+1)
#         plt.title(f'Head: {head}, Channel: 0')
#         plt.imshow(img_channel.cpu().numpy())
#     plt.suptitle(f'Image ID: {img_id}')
#     plt.tight_layout()
#     plt.show()


# for img_id, img_data in data['heads'].items():
#     plt.figure(figsize=(15, 15))

#     # Display the original image
#     img = img_data['orig'][0]
#     plt.subplot(4, 1, 1)
#     plt.title(f'Image ID: {img_id}, Modality: {data["modality"]}, Drop_prob: {data["drop_prob"]}, Bayes_n: {data["bayes_n"]}')
#     plt.imshow(img.cpu().numpy())
    
#     # Display the first channel of heads['heatmap']
#     heatmap = img_data['heatmap'][0]
#     plt.subplot(4, 1, 2)
#     plt.title(f'Head: heatmap, Channel: 0')
#     plt.imshow(heatmap.cpu().numpy())
    
#     # Display the first channel of outs
#     outs = data['outs'][img_id][0].cpu()
#     # print(torch.relu(outs))
#     # outs = torch.permute(outs, (1, 2, 0))
#     plt.subplot(4, 1, 3)
#     plt.title(f'Outs, Channel: 0')
#     plt.imshow(torch.relu(outs))

#     #normalise outs and show it
#     outs = (outs - outs.min()) / (outs.max() - outs.min())
#     plt.subplot(4, 1, 4)
#     plt.title(f'Outs_norm, Channel: 0')
#     plt.imshow(outs)


#     plt.tight_layout()
#     plt.show()


# Assuming data_dicts is a list of all your data dictionaries
# for img_id, img_data in data['heads'].items():
#     for data in data_dicts:
    
#         plt.figure(figsize=(15, 15))

#         # Display the original image
#         img = img_data['orig'][0]
#         plt.subplot(4, len(data_dicts), 1)
#         plt.title(f'Image ID: {img_id}, Modality: {data["modality"]}, Drop_prob: {data["drop_prob"]}, Bayes_n: {data["bayes_n"]}')
#         plt.imshow(img.cpu().numpy())
        
#         # Display the first channel of heads['heatmap']
#         heatmap = img_data['heatmap'][0]
#         plt.subplot(4, len(data_dicts), 2)
#         plt.title(f'Head: heatmap, Channel: 0')
#         plt.imshow(heatmap.cpu().numpy())
        
#         # Display the first channel of outs
#         outs = data['outs'][img_id][0].cpu()
#         plt.subplot(4, len(data_dicts), 3)
#         plt.title(f'Outs, Channel: 0')
#         plt.imshow(outs)

#     plt.tight_layout()
#     plt.show()



for img_id, img_data in data['heads'].items():
    plt.figure(figsize=(15, 10))

    for i, data in enumerate(data_dicts):
        # Display the original image
        img = img_data['orig'][0]
        plt.subplot(len(data_dicts), 3, 1+i*3)
        plt.title(f'Image ID: {img_id}, Modality: {data["modality"]}, Drop_prob: {data["drop_prob"]}, Bayes_n: {data["bayes_n"]}')
        plt.imshow(img.cpu().numpy())
        
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