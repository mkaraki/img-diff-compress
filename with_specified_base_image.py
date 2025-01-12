import cv2
import numpy as np

import img_diff


if __name__ == '__main__':
    base_img = 'test_imgs/a/002.jpg'
    diff_imgs = ['test_imgs/a/003.jpg', 'test_imgs/a/004.jpg', 'test_imgs/a/005.jpg', 'test_imgs/a/006.jpg',
                 'test_imgs/a/007.jpg', 'test_imgs/a/008.jpg']
    out_dir = 'test_img_out/a/'

    base = cv2.imread(base_img)
    base_size = img_diff.ImgDiff.get_image_size(base)
    if img_diff.ImgDiff.is_image_has_transparency(base):
        print('Base image has transparency')
        exit(1)
    total_pixels = base_size[0] * base_size[1]

    for img in diff_imgs:
        print('Loading', img)
        diff = cv2.imread(img)
        imgDiff = img_diff.ImgDiff(base, diff)

        distance = imgDiff.get_diff_distance()

        # debug: show diff_abs image:
        #cv2.imshow('diff_abs', diff_abs)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        print('Processing', img,
              'precalculated distance:', distance, 'of', total_pixels, int(distance / total_pixels * 100), '%',
              )
        diff_alpha = imgDiff.create_diff_img()

        print('Done. Verifying...')

        imgDiff.verify_diff_img(strict=False)

        cv2.imwrite(out_dir + img.split('/')[-1] + '.diff.png', diff_alpha)

    empty_base = np.zeros((base_size[0], base_size[1], 4), np.uint8)
    cv2.imwrite(out_dir + base_img.split('/')[-1] + '.diff.png', empty_base)

    cv2.imwrite(out_dir + 'base.png', base)
    print('Done')
