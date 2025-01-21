import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from distance_calculator import DistanceCalculator

class ImageClassifier:

    @staticmethod
    def classify_with_auto_clustering(img_list, full_distance_scan=False):
        target_idx = 2 if full_distance_scan else 1

        img_num = max_cluster_size = len(img_list)

        if img_num <= 1: # 1 or 0 images can not be classified
            return { 0: img_list }
        elif img_num == 2: # 2 images can be more simple to auto (manual) classify
            simple_distance = (img_list[0][target_idx].shape[0] * img_list[0][target_idx].shape[1]) * 2
            complex_distance = DistanceCalculator.calculate_two_image_distance(img_list[0], img_list[1], target_idx)
            if complex_distance < simple_distance:
                return { 0: img_list }
            else:
                return { 0: [img_list[0]], 1: [img_list[1]] }



        best_distance_sum = sys.maxsize
        best_classified_result = None

        for cluster_size in range(1, max_cluster_size + 1):
            if cluster_size == 1: # 1 cluster is not classified. Skip heavy calculation
                classified_images = { 0: img_list }
            else:
                classified_images = ImageClassifier.classify(img_list, cluster_size)

            total_distance = 0
            for cls, item in classified_images.items():
                if len(item) == 1: # 1 image can not determine distance. Take full size as distance
                    distance = item[0][target_idx].shape[0] * item[0][target_idx].shape[1]
                    total_distance += distance
                    continue

                distance_score = DistanceCalculator.calculate(item, target_idx=target_idx)
                distance = min(distance_score.values(), key=lambda x: x['sum_distance'])['sum_distance']
                total_distance += distance

            if total_distance < best_distance_sum:
                best_distance_sum = total_distance
                best_classified_result = classified_images

        return best_classified_result

    @staticmethod
    def classify(img_list, cluster_size):
        """
        Classify images in img_list

        :param: img_list: list of images to classify. Format: [(img_name, img, img_thumbnail), ...]
        :return: dict of classified images. Format: {cluster_id: [(img_name, img, img_thumbnail), ...], ...}
        """

        # Classify images
        classified_images = {}

        if len(img_list) < cluster_size:
            return { 0: img_list }

        mini_thumbs = []
        for img_path, img, im_thumb in img_list:
            mini_thumb = im_thumb.flatten()
            mini_thumbs.append(mini_thumb)

        model = KMeans(n_clusters=cluster_size)
        #model = MeanShift()
        #model = AgglomerativeClustering(n_clusters=cluster_size)

        model.fit(mini_thumbs)
        labels = model.labels_

        for i in range(len(labels)):
            label = labels[i]
            if label in classified_images:
                classified_images[label].append(img_list[i])
            else:
                classified_images[label] = [img_list[i]]

        return classified_images

    @staticmethod
    def plot_classifier_result(original_img_list, classified_images):
        """
        Plot classified images

        :param: classified_images: dict of classified images. Format: {cluster_id: [(img_name, img, img_thumbnail), ...], ...}
        """

        if len(original_img_list) == 0:
            print('No images to plot')
            return

        for cls, item in classified_images.items():
            print(f'Cluster: {cls}')
            for img_path, _, _ in item:
                print(f'  {img_path}')

        sqrt_img_num = int(np.ceil(np.sqrt(len(original_img_list))))
        fig, ax = plt.subplots(sqrt_img_num, sqrt_img_num, figsize=(10, 10))
        for i in range(len(original_img_list)):
            img = original_img_list[i]
            img_path, img, _ = img

            cluster_id = None
            for clustered_id, im_list in classified_images.items():
                for im_name, _, _ in im_list:
                    if img_path == im_name:
                        cluster_id = clustered_id
                        break
                if cluster_id is not None:
                    break
            if cluster_id is None:
                print('Image not found in classified images', img_path)
                continue

            if sqrt_img_num == 1:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(f'{cluster_id}')
                ax.axis('off')
            else:
                ax[i // sqrt_img_num, i % sqrt_img_num].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax[i // sqrt_img_num, i % sqrt_img_num].set_title(f'{cluster_id}')

        if sqrt_img_num != 1:
            for i in range(sqrt_img_num * sqrt_img_num):
                ax[i // sqrt_img_num, i % sqrt_img_num].axis('off')

        plt.show()
        plt.close()