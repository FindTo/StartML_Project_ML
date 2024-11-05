import os, pickle
from catboost import CatBoostClassifier

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model_final_proj.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model
