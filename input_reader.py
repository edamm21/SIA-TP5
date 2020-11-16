import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
with open(os.getcwd() + "/config.json") as file:
    data = json.load(file)
    excercise = data['EJ'].lower()
    font = data['FONT']
    probability = data['NOISE_PROBABILITY']
    letters_per_stage = data['LETTERS_PER_STAGE']
    minutes_per_stage = data['MINUTES_PER_STAGE']
    epochs_per_stage = data['MAX_EPOCHS_PER_STAGE']
    noise_training = data['NOISE_TRAINING']
    with_momentum = data['WITH_MOMENTUM']
    momentum = data['MOMENTUM']
    shuffling = data['ALPHABET_SHUFFLING']
    division_factor = data['DIVISION_FACTOR']
    learning_rate = data['LEARNING_RATE']

def get_json_data():
    return [excercise, font, probability, letters_per_stage, minutes_per_stage,
    epochs_per_stage, noise_training, with_momentum, momentum, shuffling, division_factor, learning_rate]
