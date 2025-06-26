## ..................... DATA Processing ..................... ## 

# Importing necessary libraries
import pandas as pd
import numpy as np
import os 
import sys
import joblib

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from src.logger import get_logger
from src.custom_exceptions import CustomException
from config.paths_config import *


## >... Initialize logger...........
logger = get_logger(__name__)


## ......... CLASS DataProcessing .......... ##

class DataProcessing:
    """
    Class for processing data.
    """

    def __init__(self, input_file , output_dir):

        # Initializing variables 
        self.input_file = input_file
        self.output_file = output_dir

        # initializing data variables we need
        self.anime_df = None
        self.ratings_df = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Initializing encoders and decoders 
        self.user2user_encoded = None
        self.user2user_decoded = None
        self.anime2anime_encoded = None
        self.anime2anime_decoded = None

        # Making the output directory if it does not exist
        os.makedirs(self.output_file, exist_ok=True)

        logger.info(f"###.................................... Data-Processing initialized .......................### ")

    
    ## Function to load the data:

    def load_data (self):
        try:
            ## load the data
            use_cols = ['user_id', 'anime_id', 'rating']
            self.ratings_df = pd.read_csv(self.input_file ,low_memory= True ,usecols= use_cols)
            logger.info('Data loaded successfully')
        except Exception as e:
            logger.error(f'Error loading data: {e}')
            raise CustomException(f'Failed to load data', e)
        

        
    def filter_users(self, min_ratings = 400):
        try:
            n_ratings = self.ratings_df['user_id'].value_counts() # Count the number of ratings per user
            self.ratings_df = self.ratings_df[self.ratings_df['user_id'].isin(n_ratings[n_ratings >= min_ratings].index)]  
            logger.info(f'Filtered users with less than {min_ratings} ratings')
        except Exception as e:
            logger.error(f'Error filtering users: {e}')
            raise CustomException(f'Failed to filter users', e)


    def scale_ratings(self):
        try:
            min_rating = self.ratings_df['rating'].min()  # Get the minimum rating
            max_rating = self.ratings_df['rating'].max()  # Get the maximum rating

            self.ratings_df['rating'] = (self.ratings_df['rating'] - min_rating) / (max_rating - min_rating)
            logger.info('Ratings scaled successfully')
        except Exception as e:
            logger.error(f'Error scaling ratings: {e}')
            raise CustomException(f'Failed to scale ratings', e)
        

    def encoding_decoding_data(self):
        try:
            ## ENCODING AND DECODING THE DATA FOR USERS:
            # Extracting all unique userids
            unique_user_ids = self.ratings_df['user_id'].unique().tolist()
            # Creating a mapping for user ids to encoded values
            self.user2user_encoded = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
            # Creating a mapping for encoded values to user ids
            self.user2user_decoded = {idx: user_id for idx, user_id in enumerate(unique_user_ids)}
            # Add encoded user ids to the dataframe
            self.ratings_df['user'] = self.ratings_df['user_id'].map(self.user2user_encoded)

            ## ENCODING AND DECODING THE DATA FOR ANIMES:
            # Extracting all unique anime ids
            unique_anime_ids = self.ratings_df['anime_id'].unique().tolist()
            # Creating a mapping for anime ids to encoded values
            self.anime2anime_encoded = {anime_id: idx for idx, anime_id in enumerate(unique_anime_ids)}
            # Creating a mapping for encoded values to anime ids
            self.anime2anime_decoded = {idx: anime_id for idx, anime_id in enumerate(unique_anime_ids)}
            # Add encoded anime ids to the dataframe
            self.ratings_df['anime'] = self.ratings_df['anime_id'].map(self.anime2anime_encoded)

            ## Shuffeling the data
            self.ratings_df = self.ratings_df.sample(frac=1, random_state=42).reset_index(drop=True)

            logger.info('Encoding and decoding of data completed successfully')


        except Exception as e:
            logger.error(f'Error encoding and decoding data: {e}')
            raise CustomException(f'Failed to encode and decode data', e)
        

    def split_data(self, test_size=1000, random_state=42):
        try:
            # Defining features and target variable
            features  = ['user', 'anime']
            target = 'rating'

            # Splitting the data into training and testing sets
            X = self.ratings_df[features].values
            y = self.ratings_df[target].values

            # Splitting the index for train and test sets
            train_indices = np.arange(len(X) - test_size)
            test_indices = np.arange(len(X) - test_size, len(X))

            # Assigning train and test sets
            X_train, X_test = X[train_indices], X[test_indices]
            self.y_train, self.y_test = y[train_indices], y[test_indices]

            ## Create seperate list to hold user and anime IDs for training and testing sets
            self.X_train = [X_train[:, 0], X_train[:, 1]]
            self.X_test = [X_test[:, 0], X_test[:, 1]]

            logger.info('Data split into training and testing sets successfully')

        except Exception as e:
            logger.error(f'Error splitting data: {e}')
            raise CustomException(f'Failed to split data', e)
        

    def save_processed_data(self):
        try:
            ## Creating an artifacts dictionary to store all processed data
            artifacts = {
                'anime2anime_encoded': self.anime2anime_encoded,
                'anime2anime_decoded': self.anime2anime_decoded,
                'user2user_encoded': self.user2user_encoded,
                'user2user_decoded': self.user2user_decoded,
                'X_train_array': self.X_train,
                'X_test_array': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test
            }

            ## Saving the artifacts dictionary data 

            for name , data in artifacts.items():
                file_path = os.path.join(self.output_file, f'{name}.pkl')
                joblib.dump(data, file_path)
                logger.info(f'Saved {name} to {file_path}')

            ## Finally saving the processed ratings dataframe
            processed_file_path = os.path.join(self.output_file, 'ratings_df.csv')
            self.ratings_df.to_csv(processed_file_path, index = False)
            logger.info(f'Saved processed ratings data to {processed_file_path}')


        except Exception as e:
            logger.error(f'Error saving processed data: {e}')
            raise CustomException(f'Failed to save processed data', e)

    

    ## .............. PROCESS ANIME & SYNOPSYS DATA .............. ##

    def process_anime_data (self):

        try:
            ## ...........Load the anime data..........##
            anime_df = pd.read_csv(ANIME_CSV, low_memory=False)
            # Change column name of MAL_ID to anime_id
            anime_df.rename(columns={'MAL_ID': 'anime_id'}, inplace=True)
            ## Replace 'Unknown' with None in the dataframe:
            anime_df.replace('Unknown', None, inplace=True)

            def get_anime_name(anime_id):
                """
                Returns the anime name based on the anime ID.

                First try to extract the English name and if not available then try to extract the Japanese name.
                """
                if anime_id in anime_df['anime_id'].values:
                    # Filter the row corresponding to the anime_id
                    anime_row = anime_df.loc[anime_df['anime_id'] == anime_id]

                    # Try to get the English name first
                    name = anime_row['English name'].values[0]
                    if not name:  # If English name is None or empty
                        name = anime_row['Name'].values[0]  # Fallback to Japanese name

                    return name

                # If the anime_id is not found in the anime DataFrame, return NaN
                return None
            
            ## Making a new column in the dataframe to store the anime names 
            # (priority to English name, if not available then Japanese name)
            anime_df['eng_version'] = anime_df['anime_id'].apply(get_anime_name)

            ## Sorting the dataframe based on the 'Score' column in descending order
            anime_df.sort_values(by  = 'Score', ascending=False, inplace=True)

            ## Filtering fewer columns only:
            cols = ['anime_id','eng_version','Score','Genres','Episodes','Type', 'Premiered','Members' ]
            anime_df = anime_df[cols]
            ## Resetting the index of the dataframe
            anime_df.reset_index(drop=True, inplace=True)

            ## Saving the processed anime data to a CSV file
            anime_file_path = os.path.join(self.output_file, 'anime_df.csv')
            anime_df.to_csv(anime_file_path, index=False)

            logger.info(f'Processed anime data saved to {anime_file_path}')


            ## ........ PROCESSING SYNOPSIS DATA .......... ##

            # Load the synopsis data
            anime_synopsis_df = pd.read_csv(SYNOPSIS_CSV, low_memory=True)

            # Change column name of MAL_ID to anime_id
            anime_synopsis_df.rename(columns={'MAL_ID': 'anime_id'}, inplace=True)

            # Specifying the columns to use from the synopsis CSV
            cols = ['anime_id', 'Name', 'Genres', 'sypnopsis']
            anime_synopsis_df = anime_synopsis_df[cols]

            # Replace 'Unknown' with None in the dataframe
            anime_synopsis_df.replace('Unknown', None, inplace=True)

            # Saving the processed synopsis data to a CSV file
            synopsis_file_path = os.path.join(self.output_file, 'anime_synopsis_df.csv')
            anime_synopsis_df.to_csv(synopsis_file_path, index=False)
            logger.info(f'Processed synopsis data saved to {synopsis_file_path}')

        except Exception as e:
            logger.error(f'Error processing anime and synopsis data: {e}')
            raise CustomException(f'Failed to process anime and synopsys data', e)
        

    ## ..... Function to run all processing steps .......... ##
    def run_preprocessing(self):
        try:
            ## Load the data
            self.load_data()

            ## Filter users with less than 400 ratings
            self.filter_users()

            ## Scale the ratings
            self.scale_ratings()

            ## Encoding and decoding the data
            self.encoding_decoding_data()

            ## Split the data into training and testing sets
            self.split_data()

            ## Save the processed data
            self.save_processed_data()

            ## Process anime and synopsis data
            self.process_anime_data()

            logger.info('Data preprocessing completed successfully')

        except Exception as e:
            logger.error(f'Error in data preprocessing: {e}')
            raise CustomException(f'Failed to run preprocessing', e)


## Function to run the data processing pipeline
if __name__ == "__main__":
    try:
        # Define input file and output directory
        input_file = ANIMELIST_CSV  # Path to the raw ratings data
        output_dir = PROCESSED_DIR   # Directory to save processed data

        # Create an instance of DataProcessing
        data_processor = DataProcessing(input_file, output_dir)

        # Run the preprocessing steps
        data_processor.run_preprocessing()

    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise CustomException(f"Main Block execution falied", e)