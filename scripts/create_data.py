# python scripts/create_data.py -i cxr-data/images/ -c cxr-data/DataEntry.csv -l cxr-data/ClassLabels.txt -o cxr-data
import pandas as pd
import os
import time
import argparse
import logging
import random
from PIL import Image
from functools import partial
from multiprocessing import Pool


# Create folder structure for Keras using Kaggle data (image folder + csv file)
# All images are contained in the image folder (image_path)
# Class labels for each image in the folder are supplied in the CSV file (csv_path)
# Only images with labels which are in the labels.txt file (label_path) will be processed
def create_training_data(image_path, csv_path, label_path, output_path):

    pool = Pool()

    try:
        dataset = pd.read_csv(csv_path)

        start = time.time()

        # Get class labels from file
        logging.info('Fetching class labels ...')
        labels = get_class_labels(label_path)
        logging.info('Detected labels: ' + ' '.join(labels))

        # Create folders
        logging.info('Creating folder structure ...')
        create_folders(labels, output_path)
        logging.info('Copying files ...')

        # Parallel execution
        pool.map(partial(copy_image, class_labels=labels, image_path=image_path, output_path=output_path),
                 dataset.itertuples(name=None))

        end = time.time()
        final_time = str(round(end - start, 4))
        logging.info('Finished in ' + final_time + 's')
    except:
        logging.exception("Exception occured")


# Get class labels from the folder as a list
def get_class_labels(label_path):
    with open(label_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


# Create folders for each class label
def create_folders(labels, output_path):

    train_path = output_path + '/train/'
    validation_path = output_path + '/validation/'
    for label in labels:
        path = train_path + label
        if not os.path.exists(path):
            os.makedirs(path)
        path = validation_path + label
        if not os.path.exists(path):
            os.makedirs(path)


# Copy image to corresponding class label folders
def copy_image(row, class_labels, image_path, output_path):

    delimiter = '|'      # Delimiter that separates class labels
    name_attr = row[1]   # File name
    label_attr = row[2]  # Class labels

    logging.info('Processing ' + name_attr)

    # Check if image has multiple class labels
    if delimiter in label_attr:
        for label in label_attr.split(delimiter):
            if label in class_labels:
                copy_image_to_folder(label, name_attr, image_path, output_path)
            else:
                logging.error('Invalid class label found for file (' + name_attr + ', ' + label + ')')
    else:
        if label_attr in class_labels:
            copy_image_to_folder(label_attr, name_attr, image_path, output_path)
        else:
            logging.error('Invalid class label found for file (' + name_attr + ', ' + label_attr + ')')


# Copy image from images folder to class label folder
# Randomly send 20% to validation folders and 80% to training
def copy_image_to_folder(label, file_name, image_path, output_path):

    dst_path = output_path
    if random.randint(0, 101) > 20:
        dst_path += '/train/' + label + "/" + file_name
    else:
        dst_path += '/validation/' + label + file_name
    src_path = image_path + file_name
    img = Image.open(src_path).convert('L').resize((448, 448), Image.ANTIALIAS)
    img.save(dst_path, "")


###############################################################

if __name__ == '__main__':

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", type=str, required=True, help="input directory of cxr images")
    ap.add_argument("-c", "--csv", type=str, required=True, help="location of csv file")
    ap.add_argument("-l", "--labels", type=str, required=True, help="location of class labels file")
    ap.add_argument("-o", "--output", type=str, required=True, help="output directory")
    args = vars(ap.parse_args())

    logging.basicConfig(filename='logfile.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]\t %(message)s')

    create_training_data(args['images'], args['csv'], args['labels'], args['output'])
