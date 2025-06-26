## .................. ..  Model Training Module .................. ##
import comet_ml
import joblib
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from src.logger import get_logger
from src.custom_exceptions import CustomException
from src.base_model import BaseModel
from config.paths_config import *

# Initialize logger
logger = get_logger(__name__)

## .................. ..  Model Training Class .................. ##
class ModelTraining:
    def __init__(self,data_path):
        self.data_path= data_path

        ## Configuring the Comet ML experiment
        self.experiment = comet_ml.Experiment(api_key= 'sxSBuEttLEdD9uvdT2BMERCyU',
                                              project_name= 'anime-recommendation-mlops',
                                               workspace= 'sarvesh-yadav-5201')
        self.experiment.set_name("Model Training Experiment")

        logger.info("Comet ML experiment initialized successfully")

        logger.info("###............................. Model Training initialized .............................###")
    
    def load_data(self):
        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            logger.info("Data loaded sucesfully for Model Trainig")
            return X_train_array,X_test_array,y_train,y_test
        except Exception as e:
            raise CustomException("Failed to load data",e)
        
    def train_model(self):
        try:
            ## Load the training and the testing data
            X_train_array, X_test_array, y_train, y_test = self.load_data()

            ## Get number of unique users and anime
            n_users = len(joblib.load(USER2USER_ENCODED))
            n_anime = len(joblib.load(ANIME2ANIME_ENCODED))

            ## Initialize the model
            model = BaseModel(config_path = CONFIG_PATH)
            recommender_model = model.RecommenderNet(n_users = n_users, n_anime = n_anime)
            logger.info("Recommender model initialized successfully")

            ## Configure the call-backs 
            start_lr = 0.005        # Initial learning rate
            min_lr = 0.0001         # Minimum learning rate
            max_lr = 0.001          # Maximum learning rate
            batch_size = 50000      # Batch size for training
            rampup_epochs = 5       # Number of epochs to gradually increase the learning rate
            sustain_epochs = 10     # Number of epochs to sustain the maximum learning rate
            exp_decay = 0.8         # Exponential decay factor for learning rate

            ## Learning Rate Scheduler Function
            def lrfn(epoch):
                if epoch < rampup_epochs:
                    lr = (max_lr - start_lr) / rampup_epochs * epoch + start_lr

                elif epoch < rampup_epochs + sustain_epochs:
                    lr = max_lr

                else:
                    lr = (max_lr - min_lr)*exp_decay**(epoch - rampup_epochs - sustain_epochs) + min_lr
                    
                return lr
            
            # Early stopping callback to stop training if validation loss does not improve
            lr_callback = LearningRateScheduler(lrfn)  # Learning rate scheduler callback

            # Check if the model directory exists, if not create it

            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            os.makedirs(CHECKPOINT_FILE_PATH, exist_ok=True)

            model_checkpoint = ModelCheckpoint(
                filepath=os.path.join(CHECKPOINT_FILE_PATH, 'weights.weights.h5'),
                save_weights_only=True,
                monitor='val_loss', 
                save_best_only=True, 
                mode='min', 
                verbose=1
            )  # Model checkpoint callback to save the best model weights

            # Early stopping callback to stop training if validation loss does not improve
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                verbose=1, 
                mode='min', 
                restore_best_weights=True
            )  # Early stopping callback to stop training if validation loss does not improve

            # List of callbacks to be used during training
            my_callbacks = [ model_checkpoint, lr_callback, early_stopping]  


            logger.info("Callbacks configured successfully")

            

            ## Train the model
            try:
                history = recommender_model.fit(
                                    x=X_train_array, 
                                    y=y_train, 
                                    batch_size=batch_size, 
                                    epochs=20, 
                                    verbose=1,
                                    validation_data = (X_test_array, y_test), 
                                    callbacks=my_callbacks
                                )
                
                ## Loading the model weights 
                recommender_model.load_weights(os.path.join(CHECKPOINT_FILE_PATH, 'weights.weights.h5'))

                logger.info("Model training completed successfully")

            except Exception as e:
                logger.error(f"Error during model training: {e}")
                raise CustomException('Error during model training', e)
            
            ## SAVING THE MODELS and WEIGHTS 
            self.save_model_weights(recommender_model)

            ## Logging the training history to Comet ML
            for epoch in range(len(history.history['loss'])):
                train_loss = history.history["loss"][epoch]
                val_loss = history.history["val_loss"][epoch]

                self.experiment.log_metric('train_loss', train_loss, step=epoch)
                self.experiment.log_metric('val_loss', val_loss, step=epoch)
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise CustomException('Error Training model', e)

    ## Save the model and weights 
    def save_model_weights (self, model):
        try:

            # Save the model architecture
            model.save(MODEL_FILE_PATH)
            logger.info(f"Model saved at {MODEL_FILE_PATH}")

            ## Extract MODEL WEIGHTS: 
            def extract_weights(name, model):
                """
                Extracts the weights from the trained model.
                
                """
                try:
                    # Retrieve the specified layer from the model
                    weight_layer = model.get_layer(name)
                    # Extract the weights of the layer
                    weight = weight_layer.get_weights()[0]
                    # Normalize the weights using L2 norm
                    weight = weight / np.linalg.norm(weight, axis=1).reshape((-1, 1))
                    return weight
                except Exception as e:
                    logger.error(f"Error extracting weights from layer {name}: {e}")
                    raise CustomException(f'Error extracting weights from layer {name}', e)
            
            ## anime_weights = extract_weights('anime_embedding', model)
            anime_weights = extract_weights('anime_embedding', model)

            ## user_weights = extract_weights('user_embedding', model)
            user_weights = extract_weights('user_embedding', model)

            # Save user weights
            joblib.dump(user_weights, USER_WEIGHTS_FILE_PATH)
            logger.info(f"User weights saved at {USER_WEIGHTS_FILE_PATH}")

            # Save anime weights
            joblib.dump(anime_weights, ANIME_WEIGHTS_FILE_PATH)
            logger.info(f"Anime weights saved at {ANIME_WEIGHTS_FILE_PATH}")

            logger.info("Model and weights saved successfully")

            ## Log the model and weights to Comet ML
            self.experiment.log_asset(MODEL_FILE_PATH)
            self.experiment.log_asset(USER_WEIGHTS_FILE_PATH)
            self.experiment.log_asset(ANIME_WEIGHTS_FILE_PATH)

            logger.info("Model and weights logged to Comet ML successfully")

        except Exception as e:
            logger.error(f"Error saving model and weights: {e}")
            raise CustomException('Error Saving model and weights', e)
        

if __name__=="__main__":
    try:
        # Initialize the ModelTraining class with the data path
        model_training = ModelTraining(data_path=PROCESSED_DIR)

        # Train the model
        model_training.train_model()

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise CustomException('Error in model training', e)



        


