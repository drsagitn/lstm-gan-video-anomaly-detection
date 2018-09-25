import os

def is_existing(model_name):
    for file in os.listdir("models"):
        if model_name in file:
            return True
    return False
