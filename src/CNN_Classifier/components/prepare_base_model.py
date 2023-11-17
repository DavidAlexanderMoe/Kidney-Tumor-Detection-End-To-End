import os
import urllib.request as request
import tensorflow as tf
from zipfile import ZipFile
from pathlib import Path
from CNN_Classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    

    # define vgg-16 CNN model
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    # build full model
    # add function for custom layers and to edit model
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # if freeze_all = True -> don't train any layer
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        
        # if you want to freeze the first n layers, you freeze them and proceed
        # to update weights for the remaining not frozen layers
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        # flatten operation layer
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # custom 2 class layer for prediction
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # define full model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate = self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)


    
    # save_model is not defined -> define it as static function
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    # A static method belongs to the class rather than an instance of the class.
    # It can be called on the class itself, not on an instance of the class.
