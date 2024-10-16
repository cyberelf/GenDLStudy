import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from gendlstudy.gpt.model import TextGenerator, build_model, load_model_from_file
from model_ana import model_analysis


gpt = models.load_model("models/gpt_4tb.keras")
ana = model_analysis(gpt)

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

print(json.dumps(ana, indent=2, default=convert_to_serializable))
