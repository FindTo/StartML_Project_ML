import os
import torch
from learn_model import create_nn_to_classify
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path

    return MODEL_PATH

def load_models():
    model_path = get_model_path("nn_estinmate_likes_200xPCA_embedds_1024k_825_drop_03_02.pt")
    model = create_nn_to_classify()

    model.load_state_dict(torch.load(model_path,
                    map_location=torch.device('cpu')))
    model.eval()

    return model
