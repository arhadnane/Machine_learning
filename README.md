# Neural Network for Digit Recognition

This project implements a simple feed-forward neural network for recognizing digits (0-9) from image data using backpropagation. The neural network is trained to identify digits based on pre-processed image data, which is filtered and resized before being used to train the network. The images are in grayscale and are converted to binary values (0 or 1).

### Features

- **Neural Network Implementation**: A basic feed-forward neural network with an input layer, a hidden layer, and an output layer.
- **Sigmoid Activation**: The network uses the tanh (hyperbolic tangent) function as the activation function for both hidden and output layers.
- **Backpropagation**: Implements backpropagation to update the weights during training, with momentum and learning rate parameters for optimization.
- **Image Preprocessing**: The project includes image processing to convert digit images into a usable format for training the neural network.

### Project Structure

- `nn.py`: The Python script that contains the implementation of the neural network, training functions, and the image preprocessing logic.
- `number/`: A directory containing images of digits (0-9) in `.jpg` format.

### Dependencies

- `numpy`: For numerical operations and matrix manipulations.
- `Pillow`: For image processing (resizing, filtering, and converting to numpy arrays).
- `math`: For mathematical functions, such as the sigmoid and tanh functions.

Install dependencies using pip:

```bash
pip install numpy pillow
```

### Functions

- **`rand(a, b)`**: Generates a random number between `a` and `b`.
- **`makeMatrix(I, J, fill=0.0)`**: Creates a matrix with dimensions `I x J` and fills it with the specified value (default is 0.0).
- **`sigmoid(x)`**: Computes the sigmoid (tanh) of the input `x`.
- **`dsigmoid(y)`**: Computes the derivative of the sigmoid function.
- **`traite_img()`**: Prepares the digit images for training by resizing, converting to grayscale, flattening, and normalizing them.
- **`NN` class**: The neural network class, which includes methods for updating activations, training using backpropagation, testing, and verifying predictions.

### How It Works

1. **Image Preprocessing**: Images of digits (0-9) are loaded from the `number/` directory, resized to 10x10 pixels, and converted to binary (0 or 1) values.
2. **Network Architecture**: The neural network consists of:
   - 100 input nodes (corresponding to the 100 pixels of the image).
   - 10 hidden nodes.
   - 4 output nodes (for each digit from 0-9).
3. **Training**: The network is trained using a set of labeled patterns. Each pattern consists of an image's pixel values as input and the corresponding digit label as output.
4. **Testing**: After training, the network's performance is evaluated on the same set of patterns.
5. **Verification**: The network is also tested with individual digit images to predict the digit it recognizes.

### Usage

1. Place digit images (0-9) in the `number/` folder.
2. Run the script:

```bash
python nn.py
```

This will:
- Train the network using the provided images.
- Test the network on the same set of patterns.
- Print the network's predictions for each digit.

### Example Output

```
errorTrain 0.01000
errorTrain 0.00500
...
*************************TEST******************************
[0] -> [0.021, 0.951, 0.125, 0.020]
[1] -> [0.100, 0.900, 0.212, 0.019]
...
***********************************************************
```

### Customizing

- **Number of Hidden Nodes**: You can adjust the number of hidden nodes by modifying the `NN` class initialization.
- **Learning Rate**: The learning rate `N` and momentum factor `M` can be adjusted during training to optimize performance.
- **Training Iterations**: The number of iterations for training can also be customized to achieve better accuracy.

### Contributing

Feel free to fork the repository, submit issues, and create pull requests. Contributions to improve the model, add more features, or enhance the image preprocessing are always welcome!

### License

This project is licensed under the MIT License 
