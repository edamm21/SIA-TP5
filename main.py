from input_reader import get_json_data
from Ej1.data.font import get_font, print_letters
from Ej1.basic_autoencoder import BasicAutoencoder

params = get_json_data()
if params[0] == '1a':
    font, symbols = get_font(params[1])
    ba = BasicAutoencoder(font, 500, denoising=False, probability=params[2])
    ba.progressive_train(5, 2)
    font, symbols = get_font(params[1])
    ba.graph(font, symbols)
    ba.test(font)

elif params[0] == '1b':
    font, symbols = get_font(params[1])
    ba = BasicAutoencoder(font, 1, denoising=True, probability=params[2])
    ba.train()
    font, symbols = get_font(params[1])
    ba.graph(font, symbols)
    ba.test(font)
    #do something
    print("1b")
elif params[0] == '2':
    font, symbols = get_font(4)
    ba = BasicAutoencoder(font, 500, denoising=False, probability=params[2])
    ba.progressive_train(5, 2)
    font, symbols = get_font(4)
    ba.graph(font, symbols)
    ba.test(font)
    print("\nEXPERIMENTACIÓN")
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            ba.decode(x, y)
else:
    print("Error seleccionando ejercicio")
    exit()
