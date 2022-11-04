import os

PROJECT_PARENT_DIR = os.path.dirname(__file__)

DATA_PREPARATION_DIR = os.path.join(PROJECT_PARENT_DIR, "data_preparation")
DL_MODELS_DIR = os.path.join(PROJECT_PARENT_DIR, "dl_models")
IMAGES_DIR = os.path.join(PROJECT_PARENT_DIR, "images")
PRODUCTION_DIR = os.path.join(PROJECT_PARENT_DIR, "production")
SCRAPING_DIR = os.path.join(PROJECT_PARENT_DIR, "scraping")
SOURCE_DATA_DIR = os.path.join(PROJECT_PARENT_DIR, "source_data")

IMAGES_RAW_DIR = os.path.join(PROJECT_PARENT_DIR, r"images\raw")
IMAGES_TRANSFORMED_DIR = os.path.join(PROJECT_PARENT_DIR, r"images\transformed")
IMAGES_ADDITIONAL_DIR = os.path.join(PROJECT_PARENT_DIR, r"images\additional")
PRODUCTION_RAW_DIR = os.path.join(PRODUCTION_DIR, "raw_images")
PRODUCTION_TRANSFORMED_DIR = os.path.join(PRODUCTION_DIR, "transformed")

IMG_SIZE = 200

AGE_RANGES = {0: "20-27",
              1: "28-35",
              2: "36-44",
              3: "45-55",
              4: "56-65",
              5: "66<="}

SEPARATOR = "-" * 100
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
    "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
]

# C:\Users\*yourname*\AppData\Local\Temp\CUDA
