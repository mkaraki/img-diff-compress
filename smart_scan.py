import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.mixture import GaussianMixture

from cluster_image import ImageClassifier
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

        im_thumb = cv2.resize(im, (10, 10))

        img_size = get_image_size(im)
        if img_size in img_pixel_map:
            img_pixel_map[img_size].append((img, im, im_thumb))
        else:
            img_pixel_map[img_size] = [
                (img, im, im_thumb)
            ]
    return img_pixel_map

def get_images_in_directory(directory):
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if f.endswith('.png')]

if __name__ == '__main__':
    directory = 'test_imgs/d/'
    out_directory = 'test_img_out/d/'
    only_calc_clustering = True

    images_in_directory = get_images_in_directory(directory)
    pixel_info = get_image_pixel_map(images_in_directory)

    img_size_and_cluster_map = {}

    for img_size, img_list in pixel_info.items():
        print("Img size:", img_size)
        #classify_result = ImageClassifier.classify(img_list, cluster_size=2)
        classify_result = ImageClassifier.classify_with_auto_clustering(img_list, full_distance_scan=False)

        ImageClassifier.plot_classifier_result(img_list, classify_result)

    if not only_calc_clustering:
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
        with open(out_directory + f'/__img_diff_compressed_directory.json', 'w', encoding='utf-8') as f:
            json.dump(diff_relation_data, f, indent=4)
