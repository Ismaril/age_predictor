import os

PROJECT_PARENT_DIR = os.path.dirname(__file__)

DATA_PREPARATION_DIR = os.path.join(PROJECT_PARENT_DIR, "data_preparation")
DL_MODELS_DIR = os.path.join(PROJECT_PARENT_DIR, "dl_models")
IMAGES_DIR = os.path.join(PROJECT_PARENT_DIR, "images")
IMAGES_RAW_DIR = os.path.join(PROJECT_PARENT_DIR, r"images\raw")
IMAGES_TRANSFORMED_DIR = os.path.join(PROJECT_PARENT_DIR, r"images\transformed")
IMAGES_ADDITIONAL_DIR = os.path.join(PROJECT_PARENT_DIR, r"images\additional")
PRODUCTION_DIR = os.path.join(PROJECT_PARENT_DIR, "production")
SCRAPING_DIR = os.path.join(PROJECT_PARENT_DIR, "scraping")
SOURCE_DATA_DIR = os.path.join(PROJECT_PARENT_DIR, "source_data")

print(IMAGES_TRANSFORMED_DIR)
