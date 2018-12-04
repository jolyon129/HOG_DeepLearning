import numpy as np
import sys
import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class HOG:
    def __init__(self):
        self.img = None
        self.gray_img = None
        self.file_name = None
        pass

    def read_image(self, img_path):
        head, tail = os.path.split(img_path)
        self.file_name = tail
        img_arr = mpimg.imread(img_path)
        width, height = img_arr.shape[0], img_arr.shape[1]
        gray_arr = np.zeros((width, height), dtype='uint8')
        for i in range(width):
            for j in range(height):
                gray_arr[i][j] = round(0.299 * img_arr[i][j][0] + 0.587 * img_arr[i][j][1] + 0.114 * img_arr[i][j][2])
        plt.imshow(gray_arr, cmap='gray', vmin=0, vmax=255)
        self.gray_img = gray_arr

    def process_image(self):
        Gx, Gy, Magnitude = self.gradient_operator()
        gradient_angle = self.gradient_angle(Gx, Gy, Magnitude)

    def caculate_HOG(self):
        pass

    def __gradient_operator(self):
        '''
        Caculate the gradients, Gx and Gy, and the magnitude
        :param img: orignal image
        :return: return there values (Gx, Gy, Magnitude)
        '''
        img = self.gray_img
        # img.shape store the number of rows and columns
        width = img.shape[0]
        height = img.shape[1]
        # Initiate three Img Arrays of zero, Gx, Gy, Magnitude
        gx = np.zeros([width, height], dtype=np.uint8)
        gy = np.zeros([width, height], dtype=np.uint8)
        magnitude_arr = np.zeros([width, height], dtype=np.uint8)
        prewitt_mask = {
            'Gx': ([-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]),
            'Gy': ([1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1])
        }
        # When iterate the pixels, track the maximum and minimum of Gx and Gy
        gx_min, gy_min, gx_max, gy_max = sys.maxsize, sys.maxsize, 0, 0
        for row in range(width):
            for col in range(height):
                # if the current pixel is not out of boundary
                if 1 <= row < width - 1 and 1 <= col < height - 1:
                    sum_gx, sum_gy, new_i, new_j = 0, 0, 0, 0
                    # Convolution
                    for i in range(3):
                        for j in range(3):
                            # offset the current index
                            new_i = row + (i - 1)
                            new_j = col + (j - 1)
                            sum_gx += img[new_i][new_j] * prewitt_mask['Gx'][i][j]
                            sum_gy += img[new_i][new_j] * prewitt_mask['Gy'][i][j]
                    # absolute the value
                    gx[row][col] = abs(sum_gx)
                    gy[row][col] = abs(sum_gy)
                    #  track the the maximum and minimum of Gx and Gy, which are used in normalization
                    if gx[row][col] > gx_max:
                        gx_max = gx[row][col]
                    if gx[row][col] < gx_min:
                        gx_min = gx[row][col]
                    if gy[row][col] > gy_max:
                        gy_max = gy[row][col]
                    if gy[row][col] < gy_min:
                        gy_min = gy[row][col]

        # Track the maximum and minimum of magnitude
        mag_max, mag_min = 0, sys.maxsize
        # normalize Gx and Gy
        # and generate magnitude array
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # normalize the gradient.
                # Using the formula from wiki.
                # https://en.wikipedia.org/wiki/Normalization_(image_processing)
                gx[i][j] = (gx[i][j] - gx_min) * 255 / (gx_max - gx_min)
                gy[i][j] = (gy[i][j] - gy_min) * 255 / (gy_max - gy_min)
                magnitude_arr[i][j] = np.sqrt(np.power(gx[i][j], 2) + np.power(gy[i][j], 2))
                # Tracking the minimum and maximum of magnitude value
                if magnitude_arr[i][j] > mag_max:
                    mag_max = magnitude_arr[i][j]
                if magnitude_arr[i][j] < mag_min:
                    mag_min = magnitude_arr[i][j]

        # normalize magnitude
        for i in range(magnitude_arr.shape[0]):
            for j in range(magnitude_arr.shape[1]):
                # Using the formula from wiki.
                # https://en.wikipedia.org/wiki/Normalization_(image_processing)
                magnitude_arr[i][j] = (magnitude_arr[i][j] - mag_min) * 255 / (mag_max - mag_min)

        return gx, gy, magnitude_arr

    def __gradient_angle(self, gx, gy, magnitude):
        '''
        Caculate the array of gradient angle
        :param gx:
        :param gy:
        :param magnitude:
        :return: an array of gradient angle.
        '''
        # magnitude.shape store the number of rows and columns
        width = magnitude.shape[0]
        height = magnitude.shape[1]
        # a new array of zero
        gradient_angle = np.zeros([width, height], dtype=np.float)
        pi = np.pi
        # Each sector occupies two section, where a section is pi/8
        sec = pi / 8
        for i in range(width):
            for j in range(height):
                #  if gx=0 and gy!=0, angle = pi/2
                if gx[i][j] == 0 and gy[i][j] != 0:
                    angle = pi / 2
                elif gx[i][j] == 0 and gy[i][j] == 0:
                    gradient_angle[i][j] = 0
                else:
                    angle = np.arctan(gy[i][j] / gx[i][j])
                    if angle < 0:
                        angle = angle + 2 * pi
                    gradient_angle[i][j] = angle
        return gradient_angle


if __name__ == '__main__':
    positive_trainsets_path_arr = glob.glob('Human/Train_Positive/*.bmp')
    negative_trainsets_path_arr = glob.glob('Human/Train_Negative/*.bmp')
    hog = HOG()
    hog.read_image('Human/Train_Positive/crop001030c.bmp')
    hog.process_image()
    plt.show()
