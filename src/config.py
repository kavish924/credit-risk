import os

from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV")
DATA_PATH = os.getenv("DATA_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
MLFLOW_URI = "postgresql+psycopg2://postgres:postgres123@localhost:5432/mlflow_db"