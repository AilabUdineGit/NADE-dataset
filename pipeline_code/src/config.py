from enum import Enum

class Config(Enum):
    MODELS_TO_EVALUATE      = ["<Insert model predictions here>"]
    DATASETS_TO_EVALUATE    = ["<Insert datated evaluated here>"]
    TASK                    = "negated"
    NEG_MODEL               = "bertneg"
    PATH_INPUT              = "../datasets/input"
    PATH_OUTPUT             = "../datasets/output"
    BIOSCOPE                = "../datasets/bioscope"
    TO_EVALUATE_PATH        = "to_evaluate"
    MODEL_PREDS              = "model_predictions"
    LOG                     = False