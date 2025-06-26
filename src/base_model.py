## ............. MODEL ARCHITECTURE.......##

# Import necessary libraries

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, 
    Activation, Embedding, Dot, Flatten
)

from utils.common_functions import read_yaml
from src.custom_exceptions import CustomException
from src.logger import get_logger

##################>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## initialize logger
logger = get_logger(__name__)

## CLASS FOR BASE MODEL

class BaseModel:

    def __init__(self, config_path):
        """
        Initialize the BaseModel with configuration settings.
        
        Parameters:
        config_path (str): Path to the configuration YAML file.
        """
        try:
            self.config = read_yaml(config_path)
            logger.info("BaseModel initialized with configuration settings.")
        except Exception as e:

            logger.error(f"Error initializing BaseModel: {e}")
            raise CustomException('Error initializing BaseModel', e)
        
    ## Build the recommender model
    def RecommenderNet(self, n_users , n_anime):
        
        try:
            # embedding dimensions
            embedding_dim = self.config['model']['embedding_size']

            ## defining the input for users and anime
            user = Input(shape=(1,), name='user')
            anime = Input(shape=(1,), name='anime')

            ## defining the embedding layers for users and anime
            user_embedding = Embedding(name = 'user_embedding', input_dim=n_users, output_dim=embedding_dim)(user)
            anime_embedding = Embedding(name = 'anime_embedding', input_dim=n_anime, output_dim=embedding_dim)(anime)

            ## Making the DOT layer to calculate the dot product of user and anime embeddings (signifies similarity between user and anime)
            dot_product = Dot(name = 'dot_product', normalize= True ,axes=2)([user_embedding, anime_embedding])

            ## Flattening the dot product output
            flatten = Flatten(name='flatten')(dot_product)

            ## Adding a dense layer with ReLU activation
            dense = Dense(1, kernel_initializer= 'he_normal')(flatten)
            dense = BatchNormalization(name='batch_norm')(dense)
            dense = Activation('sigmoid')(dense)

            ## Creating the model
            model = Model(inputs=[user, anime], outputs=dense)

            optimizer = self.config['model']['optimizer']
            loss = self.config['model']['loss']
            metrics = self.config['model']['metrics']

            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


            logger.info("RecommenderNet model built successfully.")

            return model
        
        except Exception as e:
            logger.error(f"Error building RecommenderNet: {e}")
            raise CustomException('Error building RecommenderNet', e)


