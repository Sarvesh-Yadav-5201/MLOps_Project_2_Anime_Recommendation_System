## ............... COMPLETE PIPELINE ...............## 

## DATA INGESTION -> DATA PREPROCESSING -> ENCODING & DECODING -> MODEL TRAINING -> MODEL EVALUATION

from src.model_training import ModelTraining
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from utils.common_functions import read_yaml
from config.paths_config  import *


'''
REMOVING DATA INGESTION FROM THE PIPELINE. 
The data ingestion step is not required as the data is already available in the specified paths.

ALSO, this data ingestion pipeline is downloading data from the GCP bucket and saving it in our local environment.
However, once we push the data to the GitHub repository, and using DVC, there is no need to download the data here in local environment.

'''

# ## .......................... DATA INGESTION ..................................##
# config = read_yaml(CONFIG_PATH)
# data_ingestion = DataIngestion(config)
# data_ingestion.run()



## ........................... DATA PREPROCESSING ..............................##
data_processor = DataProcessing(ANIMELIST_CSV, PROCESSED_DIR)
data_processor.run_preprocessing()

## ............................. MODEL TRAINING ................................##
model_training = ModelTraining(data_path=PROCESSED_DIR)
model_training.train_model()



