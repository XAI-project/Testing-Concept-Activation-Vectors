DATA_PATH = "./data"

GRAPH_PATH = DATA_PATH + "/graphs"

BALLS_PATH = DATA_PATH + "/balls"
BALLS_TRAIN_PATH = BALLS_PATH + "/train"
BALLS_TEST_PATH = BALLS_PATH + "/test"
BALLS_VALID_PATH = BALLS_PATH + "/valid"

BASKETBALL_TRAIN_PATH = BALLS_TRAIN_PATH + "/basketball"
BASKETBALL_TEST_PATH = BALLS_TEST_PATH + "/basketball"
BASKETBALL_VALID_PATH = BALLS_VALID_PATH + "/basketball"

WEATHER_PATH = DATA_PATH + "/weather"

COLORS_PATH = DATA_PATH + "/colors"
ORANGE_PATH = COLORS_PATH + "/orange"

RANDOM_IMAGES_PATH = DATA_PATH + "/random"
RANDOM_2_IMAGES_PATH = DATA_PATH + "/random2"

CIRCLES_IMAGES_PATH = DATA_PATH + "/circles"
CIRCLES_FILLED_IMAGES_PATH = DATA_PATH + "/circles_filled"

PARQUET_IMAGES_PATH = DATA_PATH + "/parquet"

MODEL_PATH = DATA_PATH + "/model.pt"


BALLS_CLASSES = [
    "basketball",
    "bowling ball",
    "brass",
    "soccer ball",
    "volley ball",
    "water polo ball",
    #"bowling ball",
    #"golf ball",
]

WEATHER_CLASSES = ["snow", "sandstorm", "dew", "lightning", "rainbow"]
