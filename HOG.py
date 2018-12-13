import numpy as np
import sys
import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import imageio
import math


class HOG:
    def __init__(self):
        self.img = None
        self.gray_img = None
        self.file_name = None
        self.cells = None
        self.blocks = None
        self.descriptor = None
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
        gx, gy, magnitude = self.__gradient_operator()
        gradient_angle = self.__gradient_angle(gx, gy, magnitude)
        self.gradient_angle = gradient_angle
        self.magnitude = magnitude
        self.caculate_HOG(gradient_angle, magnitude)
        self.normalize_over_blocks()
        self.generate_descriptor()

    def save_files(self, path1, path_of_magnitude_imgs):
        file = open(os.path.join(path1, self.file_name[:-4]) + '.text', 'w')
        for i in range(len(self.descriptor)):
            if i != len(self.descriptor) - 1:
                file.write(str(self.descriptor[i]) + '\n')
            else:
                file.write(str(self.descriptor[i]))
        file.close()

        np.save(os.path.join(path1, self.file_name[:-4]), self.descriptor)

        imageio.imwrite(os.path.join(path_of_magnitude_imgs, self.file_name), self.magnitude)

    def generate_descriptor(self):
        '''
        Concatenate vectors from all blocks to form the final descriptor
        :return: The final descriptor. The dimension of descriptor is 7524
        '''
        blocks = self.blocks
        descriptor = []
        for i in range(len(blocks)):
            for j in range(len(blocks[0])):
                descriptor += blocks[i][j]
        self.descriptor = descriptor
        return self.descriptor

    def caculate_HOG(self, gradient_angle, magnitude):
        height, width = gradient_angle.shape[0], gradient_angle.shape[1]
        cell_size = 8
        # The 2-d array of all cells, each cell is 8*8 pixels
        # each element cells[i][j] is the histogram of cells[i][j], which is a list
        self.cells = [[None] * int(width / cell_size) for i in range(int(height / cell_size))]
        for i in range(len(self.cells)):
            for j in range(len(self.cells[0])):
                self.cells[i][j] = self.hist_per_cell(gradient_angle, magnitude, [i, j], cell_size)
        return self.cells

    def normalize_over_blocks(self):
        '''
        normalize the vectors over blocks
        :return: return the matrix of blocks, each element of the matrix, blocks, contains the HOG feature.
        Each feature is a 36*1 vector
        '''
        # each block is 2*2 cells, which is 16*16 pixels
        blocks = [[None] * 11 for i in range(19)]
        # block_size = 2
        for i in range(len(blocks)):
            for j in range(len(blocks[0])):
                # Find the corresponding cells which belongs to the current block
                # For a block blocks[i][j], it occupies the cells whose row and column number satisfy the following,
                # cell[m][n] where i<=m<i+2, j<=n<j+2
                row_range = (i, i + 2)
                col_range = (j, j + 2)
                new_vec = []
                for m in range(*row_range):
                    for n in range(*col_range):
                        # Concatenate histograms from the 4 cells  to form a long vector
                        new_vec += self.cells[m][n]
                # Calculate L2 norm
                temp = 0
                for x in new_vec:
                    temp += x * x
                temp = math.sqrt(temp)
                # Normalize vector
                if temp != 0:
                    for v in range(len(new_vec)):
                        new_vec[v] = new_vec[v] / temp
                blocks[i][j] = new_vec
        self.blocks = blocks
        return self.blocks

    def hist_per_cell(self, gradient_angle, magnitude, cell_index, cell_size):
        '''
        Caculate the local gradient orientation histograms for cells

        :param gradient_angle:  gradient_angle
        :param magnitude: magnitude
        :param cell_index: the index of current cell in the cells.
        If the index of the current cell is [i,j], then this cell should contains
        the pixels where the index of the row is [i*8,(i+1)*8) and the index of the column is [i*8,(i+1)*8) in
        the original image matrix

        :param cell_size: the size of a cell, 8*8
        :return:
        '''
        # create a new histogram where the length of bin is 9
        hist_per_cell = [0] * 9
        # the range of the index of the current cell
        # If the index of cell is [i,j], then this cell should contains
        # the pixels where the index of the row is [i*8,(i+1)*8) and the index of the column is [i*8,(i+1)*8)
        row_range = (cell_index[0] * cell_size, (cell_index[0] + 1) * cell_size)
        col_range = (cell_index[1] * cell_size, (cell_index[1] + 1) * cell_size)
        for i in range(*row_range):
            for j in range(*col_range):
                angle = gradient_angle[i][j]
                deg = np.rad2deg(angle)
                if 350 <= deg < 360:
                    deg -= 360
                if 170 <= deg < 360:
                    # If the gradient angle is in the range [170, 350) degrees,
                    # simply subtract by 180 first.
                    deg -= 180
                # the array of the 9 bin centers
                bins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
                total_vote = 20
                # If the degree is within the first bin or last bin
                if -10 <= deg < 0:
                    hist_per_cell[0] = (1 - (abs(deg - bins[0]) / total_vote)) * magnitude[i][j]
                    hist_per_cell[8] = (abs(deg - bins[0]) / total_vote) * magnitude[i][j]
                elif 160 <= deg < 170:
                    hist_per_cell[8] = (1 - ((deg - bins[8]) / total_vote)) * magnitude[i][j]
                    hist_per_cell[0] = ((deg - bins[8]) / total_vote) * magnitude[i][j]
                else:
                    # Find the closest bins where bin[k]<=degree< bin[k+1]
                    k = math.floor(deg / 20)
                    # If the degree is between two bins, split the weight into the two
                    # closest bins based on their distance to bin center
                    hist_per_cell[k] = (1 - ((deg - bins[k]) / total_vote)) * magnitude[i][j]
                    hist_per_cell[k + 1] = (1 - ((bins[k + 1] - deg) / total_vote)) * magnitude[i][j]

        return hist_per_cell

    def __gradient_operator(self):
        '''
        Caculate the gradients, Gx and Gy, and the magnitude
        :param img: orignal image
        :return: return there values (Gx, Gy, Magnitude)
        '''
        img = self.gray_img
        # img.shape store the number of rows and columns
        height = img.shape[0]
        width = img.shape[1]
        # Initiate three Img Arrays of zero, Gx, Gy, Magnitude
        gx = np.zeros([height, width])
        gy = np.zeros([height, width])
        magnitude_arr = np.zeros([height, width])
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
        for row in range(height):
            for col in range(width):
                # if the current pixel is not out of boundary
                if 1 <= row < height - 1 and 1 <= col < width - 1:
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
                    gx[row][col] = sum_gx
                    gy[row][col] = sum_gy

        # Track the maximum and minimum of magnitude
        mag_max, mag_min = 0, sys.maxsize
        # generate magnitude array
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
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
                # Round off the values into integer
                magnitude_arr[i][j] = round(magnitude_arr[i][j] - mag_min) * 255 / (mag_max - mag_min)

        return gx, gy, magnitude_arr

    def __gradient_angle(self, gx, gy, magnitude):
        hist_per_cell = '''
        Caculate the array of gradient angle
        :param gx:
        :param gy:
        :param magnitude:
        :return: an array of gradient angle. Each angle is belong to [0, 2*pi]
        '''
        # magnitude.shape store the number of rows and columns
        height = magnitude.shape[0]
        width = magnitude.shape[1]
        # a new array of zero
        gradient_angle = np.zeros([height, width], dtype=np.float)
        pi = np.pi
        # Each sector occupies two section, where a section is pi/8
        sec = pi / 8
        for i in range(height):
            for j in range(width):
                #  if gx=0 and gy!=0, angle = pi/2
                if gx[i][j] == 0 and gy[i][j] != 0:
                    angle = pi / 2
                elif gx[i][j] == 0 and gy[i][j] == 0:
                    angle = 0
                else:
                    angle = np.arctan(gy[i][j] / gx[i][j])
                if angle < 0:
                    angle = angle + 2 * pi
                gradient_angle[i][j] = angle
        return gradient_angle


if __name__ == '__main__':
    positive_train_path_arr = glob.glob('Human/Train_Positive/*.bmp')
    negative_train_path_arr = glob.glob('Human/Train_Negative/*.bmp')
    for image_path in positive_train_path_arr:
        hog = HOG()
        hog.read_image(image_path)
        hog.process_image()
        hog.save_files('stores/hog_descriptor/train_positive', 'stores/magnitude_imgs/train_positive')
    for image_path in negative_train_path_arr:
        hog = HOG()
        hog.read_image(image_path)
        hog.process_image()
        hog.save_files('stores/hog_descriptor/train_negative', 'stores/magnitude_imgs/train_negative')

    # Calculate the HOG descriptor of test sets
    positive_test_file_arr = glob.glob('Human/Test_Positive/*.bmp')
    negative_test_file_arr = glob.glob('Human/Test_Negative/*.bmp')
    for image_path in positive_test_file_arr:
        hog = HOG()
        hog.read_image(image_path)
        hog.process_image()
        hog.save_files('stores/hog_descriptor/test_positive', 'stores/magnitude_imgs/test_positive')
    for image_path in negative_test_file_arr:
        hog = HOG()
        hog.read_image(image_path)
        hog.process_image()
        hog.save_files('stores/hog_descriptor/test_negative', 'stores/magnitude_imgs/test_negative')
