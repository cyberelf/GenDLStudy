"""Utilities for model metadata analysis."""

from tensorflow.keras.models import load_model, Model
import tensorflow as tf

def load_model_from_file(model_file):
    """Load a model from a file."""
    return load_model(model_file)

def model_analysis(model: Model):
    """Analyze the metadata of a model."""
    paramter_stats = {}

    paramter_stats["model_stats"] = {
        "total_params": model.count_params(),
        "trainable_params": sum([tf.size(w).numpy() for w in model.trainable_weights]),
        "non_trainable_params": sum([tf.size(w).numpy() for w in model.non_trainable_weights]),
        "parameter_weight_mean": sum([tf.reduce_mean(w).numpy() for w in model.trainable_weights]),
        "parameter_weight_std": sum([tf.math.reduce_std(w).numpy() for w in model.trainable_weights]),
        "parameter_weight_min": sum([tf.reduce_min(w).numpy() for w in model.trainable_weights]),
        "parameter_weight_max": sum([tf.reduce_max(w).numpy() for w in model.trainable_weights]),
    }

    layer_stats = []
    for layer in model.layers:
        layer_stats.append({
            "name": layer.name,
            "class": layer.__class__.__name__,
            "trainable_params": sum([tf.size(w).numpy() for w in layer.trainable_weights]),
            "non_trainable_params": sum([tf.size(w).numpy() for w in layer.non_trainable_weights]),
            "parameter_weight_mean": sum([tf.reduce_mean(w).numpy() for w in layer.trainable_weights]),
            "parameter_weight_std": sum([tf.math.reduce_std(w).numpy() for w in layer.trainable_weights]),
            "parameter_weight_min": sum([tf.reduce_min(w).numpy() for w in layer.trainable_weights]),
            "parameter_weight_max": sum([tf.reduce_max(w).numpy() for w in layer.trainable_weights]),
        })
    
    paramter_stats["layer_stats"] = layer_stats

    return paramter_stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the metadata of a model.")
    parser.add_argument("model_file", type=str, help="The model file to analyze.")
    args = parser.parse_args()

    print(model_analysis(args.model_file))
