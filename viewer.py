import json
import matplotlib as plt
import cv2


if __name__ == '__main__':
    target_dir = 'test_img_out/d/'

    with open(target_dir + '/__img_diff_compressed_directory.json', 'r') as f:
        cluster = json.load(f)

    # Natural Sort via key
    cluster = dict(sorted(cluster.items(), key=lambda x: x[0]))

    for key, value in cluster.items():
        print('Cluster:', key)
        base_img_path = value['base_image']
        base_img = cv2.imread(target_dir + base_img_path)

        if value['diff_image'] is not None:
            diff_img_path = value['diff_image']
            diff_img = cv2.imread(target_dir + diff_img_path, -1)

            # if no alpha channel, throw error
            if diff_img.shape[2] != 4:
                print('Diff image has no alpha channel')
                raise Exception('Diff image has no alpha channel')

            for i in range(base_img.shape[0]):
                for j in range(base_img.shape[1]):
                    if diff_img[i, j][3] != 0:
                        base_img[i, j] = diff_img[i, j][:3]

        cv2.imshow('img', base_img)
        cv2.setWindowTitle('img', key)
        cv2.waitKey(0)
        cv2.destroyAllWindows()