import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path

from CNN_Classifier.entity.config_entity import TrainingConfig




class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    

    # load model from artifacts folder 
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )



    # function for creating data splits using keras in-built
    # copied and pasted from keras documentation (keras -> train valid generator)
    def train_valid_generator(self):
        
        # augmentation and rescaling operations
        datagenerator_kwargs = dict(
            rescale = 1./255,           # rescaling images to have pixel values in range [0,1]
            validation_split=0.20)      # validation test size

        # params used for data generators for both training and validation
        dataflow_kwargs = dict(
            # set the specified target size for the images = all dimensions except the last one
            # common practice for specifying the height and width of images while leaving the 
            # number of channels (RGB channels) unchanged
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear")
            # image resizing parameter -> bilinear interpolation takes a weighted average 
            # of four pixels around the target pixel to resize the image

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    

    # train and then save the model
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    # training function
    def train(self):
        # steps for image classification
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )