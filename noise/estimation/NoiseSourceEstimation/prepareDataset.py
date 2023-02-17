# Author: xxx, 2017.08.19
# Modified by xxx, 2023.02.10
from __future__ import print_function
import numpy as np
import h5py
import os
import re
from numpy.lib import stride_tricks
import cv2
import tifffile as tiff

DEBUG = False

class DatabaseCreator:
    #### input or output type
    IM_RGB = 0     # 三通道RGB图
    IM_GRAY = 1    # 二维灰度图
    # IM_BAYER = 2   # 二维Bayer格式
    # IM_BAYER3 = 4  # 三通道Bayer格式
    IM_RAW = 5     # 自定义raw格式
    IM_TIFF_16 = 6 # uint16 tiff图像

    # 输出图像的格式
    FMT_RGB = 0
    FMT_LAB = 1
    FMT_YUV = 2

    ####  parameters
    inputFolder = 'data'
    outputPath = 'train.h5'
    ext = 'jpg|tif|tiff|png'
    inImageType = IM_RGB
    outImageType = IM_GRAY
    patchSize = 128
    stride = 128
    unit_norm = 4

    inBayerType = 'gbrg'
    outBayerType = 'gbrg'

    inDataType = np.uint8


    def __init__(self, inputFolder = '.', outputPath = 'data.h5', ext = 'jpg|JPG|tif|tiff|png|bmp', inImageType = IM_RGB, \
                 outImageType = IM_GRAY, inBayerType = 'gbrg', outBayerType = 'gbrg', inDataType = np.uint8, patchSize = 128, stride = 128,
                 unit_norm=4, save_format= FMT_RGB):
        """
        initialize all parameters
        :param inputFolder:
        :param outputPath:
        :param ext:
        :param inImageType:
        :param outImageType:
        :param inBayerType:
        :param outBayerType:
        :param patchSize:
        :param stride:
        """
        self.inputFolder = inputFolder
        self.outputPath = outputPath
        self.ext = ext
        self.inImageType = inImageType
        self.outImageType = outImageType
        self.inBayerType = inBayerType
        self.outBayerType = outBayerType
        self.inDataType = inDataType
        self.patchSize = patchSize
        self.stride = stride
        self.uint_norm = unit_norm
        self.save_format = save_format

    def read_raw(self, path, height=3024, width=4032):
        """
        read self-defined raw format, 16 bit uint
        :param height:
        :param width:
        :return: the raw data
        """
        file = open(path, "rb")
        rawdata = file.read()
        image = np.zeros([height, width], dtype=np.uint16)
        cout = 0
        for i in range(height):
            for j in range(width):
                image[i, j] = ord(rawdata[cout + 1]) * 256 + ord(rawdata[cout])
                cout += 2
                #       print 'cout : ', cout
        file.close()
        return image
    
    def cutup(self, data, blck, strd):
        """
        convert three channel image data to strided patches
        :param data: image data in 3d
        :param blck: patch size in 3d
        :param strd: stride in 3d
        :return: the patches
        """
        sh = np.array(data.shape)
        blck = np.asanyarray(blck)
        strd = np.asanyarray(strd)
        nbl = (sh - blck) // strd + 1
        strides = np.r_[data.strides * strd, data.strides]
        dims = np.r_[nbl, blck]
        data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
        return data6.reshape(-1, *blck)
        #return data6.reshape(-1, blck[0], blck[1], blck[2])

    def imread(self, path, height=3024, width=4032):
        """
        read image from path
        :param path:
        :param height: only used for raw type
        :param width: only used for raw type
        :return: the loaded image
        """
        if self.inImageType == self.IM_GRAY or self.inImageType == self.IM_RGB or \
           self.inImageType == self.IM_BAYER or self.inImageType == self.IM_BAYER3:
           image = cv2.imread(path)
        elif self.inImageType == self.IM_TIFF_16:
            image = tiff.imread(path)
        elif self.inImageType == self.IM_RAW:
            image = self.read_raw(path, height, width)
        return image

    def process_image(self, image):
        """
        convert image from input type to output type
        :param image: the input image
        :return: the processed image
        """
        if self.outImageType == self.IM_GRAY:
            if self.inImageType == self.IM_RGB:
                out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif self.inImageType == self.IM_GRAY:
                out = image
            else:
                raise BaseException('Cannot covert image.' + 'Input type : ' + self.inBayerType + ' Out type : Gray')
        elif self.outImageType == self.IM_RGB:
            assert(image.shape[-1] == 3)
            out = image

        if self.save_format == self.FMT_LAB:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        elif self.save_format == self.FMT_YUV:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2YUV)

        return out


    def list2array(self, list):
        """
        convert image list to image array
        :param list:
        :return: the array
        """
        n = len(list)

        assert n > 0

        n_array = 0
        for i in range(n):
            n_array += list[i].shape[0]

        assert n_array > 0

        array = np.zeros([n_array, list[0].shape[1], list[0].shape[2], list[0].shape[3]])

        idx = 0
        for i in range(n):
            array[idx:idx+list[i].shape[0], ...] = list[i]
            idx += list[i].shape[0]
        return array


    def _write_data(self, outFile, data, res_data, name, res_name, npart):
        array = self.list2array(data)
        res_array = self.list2array(res_data)

        if DEBUG:
            print('data[0].shape : ', data[0].shape, " array.shape : ", array.shape)

        print('Writing data to hdf5 file ... ')

        if npart > 0:
            final_name = '%s_%d' % (name, npart)
            final_res_name = '%s_%d' % (res_name, npart)
        else:
            final_name = name
            final_res_name = res_name

        print('name: ', final_name)
        print('res_name: ', final_res_name)

        outFile.create_dataset(final_name, data=array, compression='gzip')
        outFile.create_dataset(final_res_name, data=res_array, compression='gzip')

    def create_hdf5(self, name, res_name, part_num, max_num=1000):
        """
        create a hdf5 image database file from parameters.
        :param name: the name of image variable in the file
        :return: None
        """
        regexp = re.compile(r'.*\.(%s|%s)' % (self.ext.upper(), self.ext.lower()) )
        invalid = []
        data = []
        res_data = []
        n = 0
        npart = 0
        PART_NUM = part_num # 50 for S7, 20 for mi5s, 100 for wb
        MAX_NUM = max_num

        print('Loading images ... ')
        outFile = h5py.File(self.outputPath, "w")

        for d, dirs, files in os.walk(self.inputFolder):
            for f in files:
                if regexp.match(f):
                    #if n % 1000 == 0:
                    print('Image', n, ':', f)
                    try:
                        im = self.imread(os.path.join(d, f))
                    except IOError:
                        print('  Could not read file : ', f)
                        invalid.append(os.path.join(d, f))
                        continue

                    res_im = self.process_image(im)

                    if im.ndim == 2:
                        im = im[..., np.newaxis]
                        patches = self.cutup(im, [self.patchSize, self.patchSize, 1], [self.stride, self.stride, 1])
                    else:
                        patches = self.cutup(im, [self.patchSize, self.patchSize, im.shape[-1]], [self.stride, self.stride, im.shape[-1]])

                    data.append(patches)

                    if res_im.ndim == 2:
                        res_im = res_im[..., np.newaxis]
                        res_patches = self.cutup(res_im, [self.patchSize, self.patchSize, 1], [self.stride, self.stride, 1])

                    else:
                        res_patches = self.cutup(res_im, [self.patchSize, self.patchSize, res_im.shape[-1]], [self.stride, self.stride, res_im.shape[-1]])

                    res_data.append(res_patches)
                    n += 1

                    if n % PART_NUM == 0 and n > 0:
                        self._write_data(outFile, data, res_data, name, res_name, npart)

                        npart += 1
                        if npart * PART_NUM >= MAX_NUM:
                            break

            if npart * PART_NUM > MAX_NUM:
                break


        if n > npart * PART_NUM or (n > 0 and npart == 0):
            self._write_data(outFile, data, res_data, name, res_name, npart)
            npart += 1

        outFile.create_dataset('npart', data=npart)
        outFile.close()

        print('Total ', len(invalid), 'invalid files.')
        if len(invalid) > 0:
            print ('Invalid files : ', invalid)
        print('Create database ', name ,' finished.')
        
    def load_hdf5_v1(self, path, name):
        with h5py.File(path, 'r') as inFile:
            return inFile[name].value
    
def createDataset(dirInOut, resultFileNameOut):
    """
    Convert rgb images into gray-scale image patches of size 128x128 px.
    :param dirInOut: Directory with input images and location for output file.
    :param resultFileNameOut: Name of the result output file.
    """
    
    if os.path.exists(dirInOut):
        dc = DatabaseCreator(dirInOut, os.path.join(dirInOut, resultFileNameOut), \
                             inImageType=DatabaseCreator.IM_RGB, outImageType=DatabaseCreator.IM_GRAY, 
                             patchSize=128, stride=128)
        dc.create_hdf5('rgb', 'gray', part_num=50000)
    else:
        raise Exception("Directory", dirInOut, "does not exists.")

# Example to build an own dataset for training
# if __name__ == '__main__':
    # dirTrainImgs = r"./../../../data/training" # Directory with (yet unnoised) images.
    # createDataset(dirTrainImgs, "train.h5")