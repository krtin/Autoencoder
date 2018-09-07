"""
    Batcher processes the dataset and returns a generator for obtaining batches
    Currently this only handles mnist dataset
    Noise: Two noise are supported masking noise randomly some features are made zero
           and salt and pepper noise which randomly assigns either min or max values to random features
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#class Example:

#    def __init__(self, dec_input, target, corrupted_chunk, mask_pad):
#        self

class Batcher:

    def __init__(self, dataset, datadir, mode, batch_size, err_type, err_frac, antimode):
        self._dataset = dataset
        self._datadir = datadir
        self._mode = mode
        self._err_type = err_type
        self._err_frac = err_frac
        self._batch_size = batch_size
        self._antimode = antimode
        self.data, self.nfeatures = self.getData()



    def getBatches(self):
        batches = [_ for _ in self.genBatches()]

        return batches

    def genBatches(self):
        data = self.data
        for i in range(0, data.shape[0], self._batch_size):
            yield data[i:i+self._batch_size]

    def getData(self, updateData=False):

      if(self._dataset=='mnist'):
        mnist = input_data.read_data_sets(self._datadir, one_hot=True)
        if(self._mode=='train'):
            images = mnist.train.images
        elif(self._mode=='val'):
            images = mnist.validation.images
        elif(self._mode=='decode'):
            images = mnist.test.images
        else:
            raise Exception('Mode can only be train/val/decode')


        corrupted_images = self.corrupt_images(images)

        if(self._antimode):
            images = 1.-images

        data = list(zip(images, corrupted_images))
        if(self._mode!='decode'):
            np.random.shuffle(data)

        nfeatures = images.shape[1]

        return np.array(data), nfeatures

      elif(self._dataset=='timeseries'):
        time_steps = 2000
        x = np.linspace(0, 400, time_steps)

        y=np.sin(x)+1.0
        q = 2.0/16.0
        y = np.floor(y/q)
        datalen = 1000
        chunk_size = 50
        data = []
        for i in range(datalen):

            chunk_start = np.random.randint(low=0, high=time_steps-chunk_size)
            chunk = y[chunk_start:chunk_start+chunk_size].astype(int)

            corrupted_chunk = self.corruptseries(chunk)
            mask_pad = np.ones(chunk_size+1)
            start_id = 16
            stop_id = 17

            if(self._antimode):
                chunk = 15-chunk

            target = np.append(chunk, stop_id)
            dec_input = np.insert(chunk, 0, start_id)

            data.append((dec_input, target, corrupted_chunk, mask_pad))


        if(updateData):
            self.data = np.array(data)
            self.nfeatures = chunk_size
        else:
            return np.array(data), chunk_size

      else:
        raise Exception('Incompatible dataset')


    def corruptseries(self, seriesdata):
        seriesdata_copy = seriesdata.copy()
        noof_corr_features = np.round(self._err_frac * seriesdata.shape[0]).astype(np.int)

        #select subset of random indices
        mask = np.random.randint(0, seriesdata.shape[0], noof_corr_features)
        #create a random permutation of series data
        seriesdata_random = seriesdata.copy()
        seriesdata_random = np.random.permutation(seriesdata_random)

        for m in mask:
            seriesdata_copy[m] = seriesdata_random[m]

        return seriesdata_copy

    def corrupt_images(self, images):
        #number of corrupted features
        noof_corr_features = np.round(self._err_frac * images.shape[1]).astype(np.int)

        if self._err_type == 'masking':
            corrupted_images = self.masking_noise(images, noof_corr_features)

        elif self._err_type == 'salt_and_pepper':
            corrupted_images = self.salt_and_pepper_noise(images, noof_corr_features)

        return corrupted_images

    def masking_noise(self, X, cf):

        X_noise = X.copy()

        #number of images
        n_samples = X.shape[0]
        #number of features
        n_features = X.shape[1]

        for i in range(n_samples):
            #select features to mask for each image
            mask = np.random.randint(0, n_features, cf)
            #make the selected features zero
            for m in mask:
                X_noise[i][m] = 0.

        return X_noise


    def salt_and_pepper_noise(self, X, cf):

        X_noise = X.copy()
        n_features = X.shape[1]

        min = X.min()
        max = X.max()

        for i, sample in enumerate(X):
            mask = np.random.randint(0, n_features, cf)

            for m in mask:

                if np.random.random() < 0.5:
                    X_noise[i][m] = min
                else:
                    X_noise[i][m] = max

        return X_noise
