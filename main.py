from input_reader import get_json_data
from Ej1.data.font import get_font, print_letters
from Ej1.basic_autoencoder import BasicAutoencoder

params = get_json_data()
if params[0] == '1a':
    font = get_font(params[1])
    ba = BasicAutoencoder(font, 1000)
    ba.train()
elif params[0] == '1b':
    #do something
    print("1b")
elif params[0] == '2':
    # do something
    print("2")
else:
    print("Error seleccionando ejercicio")
    exit()
