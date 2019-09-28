# python scripts/train_classifier.py -t cxr-data/train -v cxr-data/validation -o model.h5
import matplotlib.pyplot as plt
import argparse
from resnet_model import build_resnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


def create_model(target_size=448, classes=15, learning_rate=0.0146, epochs=200, class_mode='categorical'):
    """
    Creates a ResNet CNN model
    :param target_size: Input image dimension
    :param classes: Number of classification categories
    :param learning_rate: Learning rate for the Adam optimizer
    :param epochs: Number of training epochs, used to define decay
    :param class_mode: Binary for 2 classification categories, categorical for more than 2
    :return: Compiled ResNet model
    """
    model = build_resnet(target_size, target_size, classes=classes)
    optimizer = Adam(learning_rate=learning_rate, decay=learning_rate/epochs)

    loss = 'categorical_crossentropy' if class_mode == 'categorical' else 'binary_crossentropy'
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train_network(train_dir, validation_dir, output_path, target_size=448, classes=15,
                  class_mode='categorical', color_mode='grayscale', batch_size=8, epochs=200, save_graph=True):
    """
    Train a ResNet CNN network and save the model so that it can be used to make predictions
    :param train_dir: Directory of training image data
    :param validation_dir: Directory of validation image data
    :param output_path: Output path of the mo
    :param target_size: Input image dimension
    :param classes: Number of classification categories
    :param class_mode: Binary for 2 classification categories, categorical for more than 2
    :param color_mode: Whether images are RGB or grayscale
    :param batch_size: Number of images to be processed in parallel
    :param epochs: Number of training epochs
    :param save_graph: Flag whether to save training graphs or not
    """

    # Create the model and show the architecture summary
    model = create_model(target_size, classes, class_mode=class_mode)
    model.summary()

    # The data augmentation strategy for test data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=7,
        horizontal_flip=True)

    # The data augmentation strategy for validation data, only rescaling
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode)

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode)

    # Train the model using above defined data generators
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator)

    # Save model to the output path
    model.save(output_path)

    if save_graph:

        # Plot accuracy values for training and validation data
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('accuracy.png')

        # Plot loss values for training and validation data
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('loss.png')


###############################################################

if __name__ == '__main__':

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", type=str, required=True, help="path to directory with training data")
    ap.add_argument("-v", "--validation", type=str, required=True, help="path to directory with validation data")
    ap.add_argument("-o", "--output", type=str, required=True, help="output path of the model")
    args = vars(ap.parse_args())

    train_network(args["train"], args["validation"], args["output"])
