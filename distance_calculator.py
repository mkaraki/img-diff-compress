import csv
import sys

import cv2

import img_diff


class DistanceCalculator:

    @staticmethod
    def calculate(img_list):
        """
        Calculate distance between images in img_list
        Small distance means similar images
        """

        distance_score = {}

        for i in range(len(img_list)):
            img_a = img_list[i][1]
            sum_distance = 0

            for j in range(len(img_list)):
                if i == j:
                    continue

                img_b = img_list[j][1]

                imgDiff = img_diff.ImgDiff(img_a, img_b)
                distance = imgDiff.get_diff_distance()

                sum_distance += distance

            distance_score[img_list[i][0]] = {
                'sum_distance': sum_distance,
            }

        return distance_score


if __name__ == '__main__':
    imgs = [ 'test_imgs/a/002.jpg', 'test_imgs/a/003.jpg', 'test_imgs/a/004.jpg', 'test_imgs/a/005.jpg',
                  'test_imgs/a/006.jpg', 'test_imgs/a/007.jpg', 'test_imgs/a/008.jpg']
    out_file = 'test_img_out/a/distance_report.csv'

    best_avg_value = sys.maxsize
    best_avg_img = None
    best_max_value = sys.maxsize
    best_max_img = None
    best_min_value = sys.maxsize
    best_min_img = None
    best_sum_value = sys.maxsize
    best_sum_img = None

    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img_a', 'img_b', 'distance', 'distance_percent'])

        for i in range(len(imgs)):
            img_a = cv2.imread(imgs[i])
            img_pixels = img_a.shape[0] * img_a.shape[1]
            min_distance = img_pixels
            max_distance = 0
            sum_distance = 0

            for j in range(len(imgs)):
                if i == j:
                    continue

                img_b = cv2.imread(imgs[j])

                imgDiff = img_diff.ImgDiff(img_a, img_b)
                distance = imgDiff.get_diff_distance()
                distance_percent = distance / img_pixels

                if distance < min_distance:
                    min_distance = distance
                if distance > max_distance:
                    max_distance = distance
                sum_distance += distance

                print('Processing', imgs[i], '<>', imgs[j],
                      'precalculated distance:', distance, 'of', img_pixels, int(distance_percent * 100), '%',
                      )

                writer.writerow([imgs[i], imgs[j], distance, distance_percent])

            avg_distance = sum_distance / (len(imgs) - 1)
            avg_perc = int(avg_distance / img_pixels * 100)
            print('Relation with', imgs[i],
                  'min:', min_distance, 'max:', max_distance, 'avg:', avg_distance, avg_perc, '%',
                  'sum:', sum_distance
                  )

            if avg_distance < best_avg_value:
                best_avg_value = avg_distance
                best_avg_img = imgs[i]
            if max_distance < best_max_value:
                best_max_value = max_distance
                best_max_img = imgs[i]
            if min_distance < best_min_value:
                best_min_value = min_distance
                best_min_img = imgs[i]
            if sum_distance < best_sum_value:
                best_sum_value = sum_distance
                best_sum_img = imgs[i]

    print('Best avg:', best_avg_img, best_avg_value)
    print('Best max:', best_max_img, best_max_value)
    print('Best min:', best_min_img, best_min_value)
    print('Best sum:', best_sum_img, best_sum_value)