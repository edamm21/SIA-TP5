import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
with open(os.getcwd() + "/config.json") as file:
    data = json.load(file)
    excercise = data['EJ'].lower()
    font = data['FONT']
    probability = data['NOISE_PROBABILITY']

def get_json_data():
    return [excercise, font, probability]