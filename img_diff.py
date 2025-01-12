import numpy as np
import cv2


class ImgDiff:

    def __init__(self, base_img, diff_img):
        self.diff_abs_1ch = None
        self.diff_distance = None
        self.diffed_img = None
        self.base_img = base_img
        self.diff_img = diff_img

        if self.is_image_has_transparency(self.diff_img):
            print('Diff image has transparency')
            raise Exception('Diff image has transparency')
        elif self.is_image_has_transparency(self.base_img):
            print('Base image has transparency')
            raise Exception('Base image has transparency')

        base_size = self.get_image_size(self.base_img)
        diff_size = self.get_image_size(self.diff_img)
        if base_size != diff_size:
            print('Base image and diff image size are not same')
            raise Exception('Base image and diff image size are not same')
        self.img_size = base_size

    @staticmethod
    def is_image_has_transparency(img):
        if img.shape[2] == 4:
            # Check all the alpha values are 0
            return not cv2.bitwise_not(img[:, :, 3]).any()
        return False

    @staticmethod
    def get_image_size(img):
        return img.shape[0], img.shape[1]

    def get_diff_abs_one_channel(self):
        if self.diff_abs_1ch is not None:
            return self.diff_abs_1ch

        diff_abs = cv2.absdiff(self.base_img, self.diff_img)
        # max each pixel of 3 channels
        # returns 1 channel image
        self.diff_abs_1ch = np.max(diff_abs, axis=2)
        return self.diff_abs_1ch

    def get_diff_distance(self):
        if self.diff_distance is not None:
            return self.diff_distance
        if self.diff_abs_1ch is None:
            self.get_diff_abs_one_channel()

        # Count non-zero pixels
        self.diff_distance = np.count_nonzero(self.diff_abs_1ch)
        return self.diff_distance

    def create_diff_img(self):
        if self.diffed_img is not None:
            return self.diffed_img

        diff_alpha = np.zeros((self.img_size[0], self.img_size[1], 4), np.uint8)
        if self.diff_abs_1ch is None:
            self.get_diff_abs_one_channel()

        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                if self.diff_abs_1ch[i][j] > 0:
                    diff_alpha[i, j] = [self.diff_img[i, j][0], self.diff_img[i, j][1], self.diff_img[i, j][2], 255]

        self.diffed_img = diff_alpha
        return self.diffed_img

    def verify_diff_img(self, strict=True):
        if self.diffed_img is None:
            self.create_diff_img()

        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                original_pixel = self.diff_img[i, j][:3]
                base_pixel = self.base_img[i, j][:3]
                diffed_pixel = self.diffed_img[i, j][:3]
                diffed_transparency = self.diffed_img[i, j][3]
                if diffed_transparency == 0:
                    # When diffed pixel is transparent, original and base pixel should be same
                    assert np.array_equal(original_pixel, base_pixel)
                elif diffed_transparency == 255:
                    # When diffed pixel is not transparent, original and diffed pixel should be same
                    assert np.array_equal(original_pixel, diffed_pixel)
                    # And base pixel should not be same
                    if strict:
                        assert not np.array_equal(original_pixel, base_pixel)
                else:
                    raise Exception('Diffed pixel has invalid transparency value. This should be 0 or 255.')
