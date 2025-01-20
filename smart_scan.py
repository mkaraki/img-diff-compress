import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import json

from distance_calculator import DistanceCalculator
from img_diff import ImgDiff

def get_image_size(img):
    return img.shape[0], img.shape[1]

def get_image_pixel_map(imgs):
    img_pixel_map = {}
    for img in imgs:
        im = cv2.imread(img, -1)

        if ImgDiff.is_image_has_transparency(im):
            print('Image has transparency:', img)
            continue

        img_size = get_image_size(im)
        if img_size in img_pixel_map:
            img_pixel_map[img_size].append((img, im))
        else:
            img_pixel_map[img_size] = [
                (img, im)
            ]
    return img_pixel_map

def get_images_in_directory(directory):
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if f.endswith('.png')]

if __name__ == '__main__':
    directory = 'test_imgs/d/'
    out_directory = 'test_img_out/d/'

    images_in_directory = get_images_in_directory(directory)
    pixel_info = get_image_pixel_map(images_in_directory)

    img_size_and_cluster_map = {}

    for img_size, img_list in pixel_info.items():
        print(f'Image size: {img_size}')
        cluster_size = 2

        if len(img_list) < cluster_size:
            print(f'Need at least {cluster_size} images to compare')
            img_size_and_cluster_map[(img_size[0], img_size[1], 0)] = img_list
            continue

        mini_thumbs = []
        for img_path, img in img_list:
            mini_thumb = cv2.resize(img, (10, 10))
            mini_thumb = mini_thumb.flatten()
            mini_thumbs.append(mini_thumb)

        kmeans = KMeans(n_clusters=cluster_size, init = 'k-means++', random_state=42)
        kmeans.fit_predict(mini_thumbs)

        #celling_sqrt_image_num = int(np.ceil(np.sqrt(len(img_list))))

        ## Show clustering result
        #fig, axs = plt.subplots(celling_sqrt_image_num, celling_sqrt_image_num)
        #for i, (img_path, img) in enumerate(img_list):
        #    ax = axs[i // celling_sqrt_image_num, i % celling_sqrt_image_num]
        #    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #    ax.set_title(f'{kmeans.labels_[i]}')
        #    ax.axis('off')

        ## Hide empty subplots
        #for i in range(len(img_list), celling_sqrt_image_num * celling_sqrt_image_num):
        #    axs[i // celling_sqrt_image_num, i % celling_sqrt_image_num].axis('off')

        #plt.show()

        # Add image size and cluster map
        # img_size_and_cluster_map[(width, height, cluster_id)] = [img_path, img])
        for i, (img_path, img) in enumerate(img_list):
            identifier_tuple = (img_size[0], img_size[1], int(kmeans.labels_[i]))
            if identifier_tuple in img_size_and_cluster_map:
                img_size_and_cluster_map[identifier_tuple].append((img_path, img))
            else:
                img_size_and_cluster_map[identifier_tuple] = [(img_path, img)]

    diff_relation_data = {}

    # Print map except img
    for key, value in img_size_and_cluster_map.items():
        print(f'Image size: {key[0]}x{key[1]}, Cluster: {key[2]}')
        diff_id = f'{key[0]}_{key[1]}_{key[2]}'
        for img_path, _ in value:
            print(f'  {img_path}')

        min_distance_img = value[0][0]
        if len(value) >= 2:
            distance_calc = DistanceCalculator.calculate(value)
            min_distance_img = min(distance_calc.items(), key=lambda x: x[1]['sum_distance'])[0]

        print('=> Decided base:', min_distance_img)

        # Get min distance CV2 image
        min_distance_img = [img for img in value if img[0] == min_distance_img][0]

        # Image without base image
        # Structure: (img_path, img)[...]
        diff_imgs = [img for img in value if img[0] != min_distance_img]

        for img_path, img in diff_imgs:
            print('Loading', img_path)
            file_name = img_path.split('/')[-1].split('\\')[-1]
            out_file_name = f'{file_name}.diff{diff_id}.png'

            diff_relation_data[file_name] = {
                'base_image': f'base{diff_id}.png',
                'diff_image': out_file_name,
            }

            imgDiff = ImgDiff(min_distance_img[1], img)

            diff_alpha = imgDiff.create_diff_img()
            imgDiff.verify_diff_img(strict=False)

            cv2.imwrite(out_directory + out_file_name, diff_alpha)

        # Write base image
        cv2.imwrite(out_directory + f'/base{diff_id}.png', min_distance_img[1])
        # Write base image info
        min_distance_img_file_name = min_distance_img[0].split('/')[-1].split('\\')[-1]
        diff_relation_data[min_distance_img_file_name] = {
            'base_image': f'base{diff_id}.png',
            'diff_image': None,
        }

        print()

    # Write diff relation data as JSON
    with open(out_directory + f'/__img_diff_compressed_directory.json', 'w') as f:
        json.dump(diff_relation_data, f, indent=4)
