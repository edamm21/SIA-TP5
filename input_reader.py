import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
with open(os.getcwd() + "/config.json") as file:
    data = json.load(file)
    excercise = data['EJ'].lower()
    font = data['FONT']
    probability = data['NOISE_PROBABILITY']
    with_momentum = data['WITH_MOMENTUM']
    momentum = data['MOMENTUM']

def get_json_data():
    return [excercise, font, probability, with_momentum, momentum]