## ..... PATHS CONFIG ..... ##
import os

####..............DATA INGESTION PATHS..............####
RAW_DIR = "artifacts/raw"
CONFIG_PATH = "config/config.yaml"


####..............DATA PREPROCESING PATHS..............####
PROCESSED_DIR = "artifacts/processed"

ANIMELIST_CSV  = "artifacts/raw/animelist.csv"
ANIME_CSV = "artifacts/raw/anime.csv"
SYNOPSIS_CSV = "artifacts/raw/anime_with_synopsis.csv"

#### ..............ENCODING & DECODING PATHS..............####
USER2USER_ENCODED = 'artifacts/processed/user2user_encoded.pkl'
USER2USER_DECODED = 'artifacts/processed/user2user_decoded.pkl'
ANIME2ANIME_ENCODED = 'artifacts/processed/anime2anime_encoded.pkl'
ANIME2ANIME_DECODED = 'artifacts/processed/anime2anime_decoded.pkl'

####..............MODEL TRAINING PATHS..............####
X_TRAIN_ARRAY = "artifacts/processed/X_train_array.pkl"
Y_TRAIN = "artifacts/processed/y_train.pkl"
X_TEST_ARRAY = "artifacts/processed/X_test_array.pkl"
Y_TEST = "artifacts/processed/y_test.pkl"


MODEL_DIR = "artifacts/model"
WEIGHTS_DIR = "artifacts/model_weights"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.h5")

ANIME_WEIGHTS_FILE_PATH = os.path.join(WEIGHTS_DIR, "anime_weights.pkl")
USER_WEIGHTS_FILE_PATH = os.path.join(WEIGHTS_DIR, "user_weights.pkl")

CHECKPOINT_FILE_PATH = 'artifacts/model_checkpoints'