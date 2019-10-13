import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from tensorflow.python.keras.utils.data_utils import Sequence
from create_data import get_class_labels


class CustomSequenceGenerator(Sequence):
    """
    Custom generator for model data
    """

    def preprocess_image(self, image, target_size):
        """
        Method used to prepare images for processing
        :param image: Image opened using io.BytesIO
        :param target_size: Target size of the image
        :return: Processed image
        """
        # Convert image to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # Resize and preprocess image
        image = image.resize((target_size, target_size))
        image = img_to_array(image)
        if self.augment is not None:
            image = self.augment(image=image)['image']
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image

    def preprocess_csv(self, csv, i):
        """
        Return feature and label data from a csv row
        :param csv: CSV file opened using pandas
        :param i: index of the CSV row
        :return: Feature and Label vectors
        """

        # Get row from csv file by index and extract features
        row = csv.loc[i, :]
        features = np.asarray([[row['Age'], row['Gender'], row['Position']]])

        # Extract labels, for multi-label data use bitwise_or to represent all classes
        labels = row['Labels']
        return features, labels

    def __init__(self, image_dir, csv_file_path, label_path, dim=448, batch_size=8,
                 n_classes=15, n_channels=1, vec_size=3, shuffle=True, augmentations=None):
        """
        Custom keras generator
        :param image_dir: Path to the image directory
        :param csv_file_path: Path to the CSV file
        :param label_path: Path to the txt file containing all labels
        :param dim: Image dimension
        :param batch_size: Batch size
        :param n_classes: Number of classes
        :param n_channels: Number of image color channels
        :param vec_size: Feature vector dimension
        :param shuffle: Defines randomness of data
        :param augmentations: Data augmentations using albumentations library
        """
        self.image_dir = image_dir
        self.image_file_list = os.listdir(image_dir)
        self.batch_size = batch_size
        self.csv_file = pd.read_csv(csv_file_path).set_index('File')
        self.n_classes = n_classes
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.vec_size = vec_size
        self.labels = get_class_labels(label_path)
        self.labels_dict = dict(zip(self.labels, range(0, len(self.labels))))
        self.augment = augmentations

    def __len__(self):
        """
        Defines data length per epoch
        """
        return int(np.ceil(len(self.image_file_list) / float(self.batch_size)))

    def __getitem__(self, index):
        """
        Generates one batch of data
        :param index: Starting index of the batch
        :return: Batch of generated data
        """

        samples = self.image_file_list[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(samples)
        return x, y

    def __data_generation(self, samples):
        """
        Creates a batch of data
        :param samples: Input samples which will be processed to create the data
        :return: Batch of generated data
        """

        # Create input and output numpy arrays
        x_batch_image = np.empty((self.batch_size, self.dim, self.dim, self.n_channels))
        x_batch_vector = np.empty((self.batch_size, self.vec_size))
        y_batch = np.empty((self.batch_size, self.n_classes))

        for i, sample in enumerate(samples):

            # Fetch each image from the file system and features/labels for that image
            image_file_path = self.image_dir + "/" + sample
            image = self.preprocess_image(Image.open(image_file_path), 448)
            features, labels = self.preprocess_csv(self.csv_file, sample)

            x_batch_image[i] = image
            x_batch_vector[i] = features
            y_batch[i] = to_categorical(self.labels_dict.get(labels), self.n_classes)

        return [x_batch_image, x_batch_vector], y_batch


###############################################################

if __name__ == '__main__':
    gen = CustomSequenceGenerator('cxr-data/images', 'cxr-data/DataEntry2.csv',
                                  'cxr-data/ClassLabels.txt', batch_size=3)
    gen.__getitem__(0)
