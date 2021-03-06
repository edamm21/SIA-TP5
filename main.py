from input_reader import get_json_data
from Ej1.data.font import get_font, print_letters
from Ej1.basic_autoencoder import BasicAutoencoder

params = get_json_data()
if params[0] == '1a':
    font, symbols = get_font(params[1])
    ba = BasicAutoencoder(font, probability=params[2], with_momentum=params[7], momentum=params[8], division_factor=params[10], learning_rate=params[11])
    ba.progressive_train(params[3], params[4], params[5], shuffling=params[9])
    font, symbols = get_font(params[1])
    ba.test(font)
    print("\nEXPERIMENTACIÓN")
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            print("En (", x, ", ", y, "):")
            ba.decode(x, y)
    ba.graph(font, symbols)

elif params[0] == '1b':
    font, symbols = get_font(params[1])
    ba = BasicAutoencoder(font, denoising=params[6], probability=params[2], with_momentum=params[7], momentum=params[8], division_factor=params[10], learning_rate=params[11])
    ba.progressive_train(params[3], params[4], params[5], shuffling=params[9])
    font, symbols = get_font(params[1])
    ba.test(font, noise=True)
    ba.graph(font, symbols)

elif params[0] == '2':
    font, symbols = get_font(4)
    ba = BasicAutoencoder(font, probability=params[2], with_momentum=params[7], momentum=params[8], division_factor=params[10], learning_rate=params[11])
    ba.progressive_train(params[3], params[4], params[5], shuffling=params[9])
    font, symbols = get_font(4)
    ba.test(font)
    print("\nEXPERIMENTACIÓN")
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            print("En (", x, ", ", y, "):")
            ba.decode(x, y)
    ba.graph(font, symbols)
else:
    print("Error seleccionando ejercicio")
    exit()
