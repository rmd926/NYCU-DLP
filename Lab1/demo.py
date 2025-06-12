import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class Generate_Data:
    @staticmethod
    def linear(n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """ Generate linearly separable data points """
        pts = np.random.uniform(0, 1, (n, 2))
        inputs = pts
        labels = (pts[:, 0] > pts[:, 1]).astype(int)
        return inputs, labels.reshape(-1, 1)

    @staticmethod
    def XOR_easy(n: int = 22) -> Tuple[np.ndarray, np.ndarray]:
        """ Generate XOR data points """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

class Activation: 
    '''
    include sigmoid, tanh, relu, leaky_relu and their derivatives
    plot their functions to visualize their behavior
    '''
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(y: np.ndarray) -> np.ndarray:
        return np.multiply(y, 1.0 - y)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def derivative_tanh(y: np.ndarray) -> np.ndarray:
        return 1.0 - y ** 2

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Calculate relu function."""
        return np.maximum(0.0, x)

    @staticmethod
    def derivative_relu(y: np.ndarray) -> np.ndarray:
        """Calculate the derivative of relu function."""
        return np.heaviside(y, 0.0)

    @staticmethod
    def leaky_relu(x: np.ndarray) -> np.ndarray:
        """Calculate leaky relu function."""
        return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)

    @staticmethod
    def derivative_leaky_relu(y: np.ndarray) -> np.ndarray:
        """Calculate the derivative of leaky relu function."""
        y[y > 0.0] = 1.0
        y[y <= 0.0] = 0.01
        return y

    print("--------------------Please Check the Activation Functions Plot--------------------")
    def plot_activation_functions():
        """Plot various activation functions to visualize their behavior."""
        x = np.linspace(-10, 10, 100)
        sigmoid = Activation.sigmoid(x)
        tanh = Activation.tanh(x)
        relu = Activation.relu(x)
        leaky_relu = Activation.leaky_relu(x)

        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(x, sigmoid, label="Sigmoid", color='red')  # Red color
        plt.title("Sigmoid Activation Function")
        plt.xlabel("x")
        plt.ylabel("Sigmoid(x)")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(x, tanh, label="Tanh", color='purple')  # Purple color
        plt.title("Tanh Activation Function")
        plt.xlabel("x")
        plt.ylabel("Tanh(x)")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(x, relu, label="ReLU", color='green')  # Green color
        plt.title("ReLU Activation Function")
        plt.xlabel("x")
        plt.ylabel("ReLU(x)")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(x, leaky_relu, label="Leaky ReLU", color='blue')  # Blue color
        plt.title("Leaky ReLU Activation Function")
        plt.xlabel("x")
        plt.ylabel("Leaky ReLU(x)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Call the function to plot the activation functions
Activation.plot_activation_functions()

class Layer:
    def __init__(self, input_links: int, output_links: int, activation: str = 'sigmoid', optimizer: str = 'gd',
                 learning_rate: float = 0.1):
        self.weight = np.random.normal(0, 1, (input_links + 1, output_links))
        self.momentum = np.zeros((input_links + 1, output_links))
        self.sum_of_squares_of_gradients = np.zeros((input_links + 1, output_links))
        self.moving_average_m = np.zeros((input_links + 1, output_links))
        self.moving_average_v = np.zeros((input_links + 1, output_links))
        self.update_times = 1
        self.forward_gradient = None
        self.backward_gradient = None
        self.output = None
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.forward_gradient = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
        if self.activation == 'sigmoid':
            self.output = Activation.sigmoid(self.forward_gradient @ self.weight)
        elif self.activation == 'tanh':
            self.output = Activation.tanh(self.forward_gradient @ self.weight)
        elif self.activation == 'relu':
            self.output = Activation.relu(self.forward_gradient @ self.weight)
        elif self.activation == 'leaky_relu':
            self.output = Activation.leaky_relu(self.forward_gradient @ self.weight)
        else:
            self.output = self.forward_gradient @ self.weight
        return self.output

    def backward(self, derivative_loss: np.ndarray) -> np.ndarray:
        # Calculate the gradient of the loss with respect to the output of the layer
        if self.activation == 'sigmoid':
            self.backward_gradient = np.multiply(Activation.derivative_sigmoid(self.output), derivative_loss)
        elif self.activation == 'tanh':
            self.backward_gradient = np.multiply(Activation.derivative_tanh(self.output), derivative_loss)
        elif self.activation == 'relu':
            self.backward_gradient = np.multiply(Activation.derivative_relu(self.output), derivative_loss)
        elif self.activation == 'leaky_relu':
            self.backward_gradient = np.multiply(Activation.derivative_leaky_relu(self.output), derivative_loss)
        else:
            self.backward_gradient = derivative_loss
        return self.backward_gradient @ self.weight[:-1].T 
    
    def update(self) -> None:
        gradient = self.forward_gradient.T @ self.backward_gradient

        if self.optimizer == 'adam':
            self.moving_average_m = 0.9 * self.moving_average_m + 0.1 * gradient
            self.moving_average_v = 0.999 * self.moving_average_v + 0.001 * np.square(gradient)
            bias_correction_m = self.moving_average_m / (1.0 - 0.9 ** self.update_times)
            bias_correction_v = self.moving_average_v / (1.0 - 0.999 ** self.update_times)
            self.update_times += 1
            delta_weight = -self.learning_rate * bias_correction_m / (np.sqrt(bias_correction_v) + 1e-8)

        elif self.optimizer == 'adagrad':
            self.sum_of_squares_of_gradients += np.square(gradient)
            delta_weight = -self.learning_rate * gradient / np.sqrt(self.sum_of_squares_of_gradients + 1e-8)

        elif self.optimizer == 'momentum':
            self.momentum = 0.9 * self.momentum - self.learning_rate * gradient
            delta_weight = self.momentum

        elif self.optimizer == 'gd':
            delta_weight = -self.learning_rate * gradient

        else: # Default to Gradient Descent
            delta_weight = -self.learning_rate * gradient

        self.weight += delta_weight

class Model:
    def __init__(self, epoch: int, learning_rate: float, num_of_hidden_layers: int, input_units: int,
                 hidden_units: int, output_units: int, activation: str, optimizer: str): 
    ## 預設2個hidden layer，每個hidden layer 會有4個hidden units，預設activation是sigmoid
        self.num_of_epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.optimizer = optimizer
        self.learning_epoch, self.learning_loss = list(), list()

        # Setup input layer
        self.input_layer = Layer(input_units, hidden_units, activation, optimizer, learning_rate)

        # Dynamically create hidden layers
        self.hidden_layers = [Layer(hidden_units, hidden_units, activation, optimizer, learning_rate)
                              for _ in range(num_of_hidden_layers)]
        # Output layer
        self.output_layer = Layer(hidden_units, output_units, 'sigmoid', optimizer, learning_rate)  # Use sigmoid for output by default

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward feed through the network
        """
        inputs = self.input_layer.forward(inputs)
        for layer in self.hidden_layers:
            inputs = layer.forward(inputs)
        return self.output_layer.forward(inputs)

    def backward(self, derivative_loss) -> None:
        
        derivative_loss = self.output_layer.backward(derivative_loss)
        for layer in reversed(self.hidden_layers):
            derivative_loss = layer.backward(derivative_loss)
        self.input_layer.backward(derivative_loss)

    def update(self) -> None:
        """
        Update weights in the entire network
        """
        self.input_layer.update()
        for layer in self.hidden_layers:
            layer.update()
        self.output_layer.update()
    
    def train(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        
        #Train the neural network with the given inputs and labels.
        for epoch in range(self.num_of_epoch):
            prediction = self.forward(inputs)
            loss = self.mse_loss(prediction=prediction, ground_truth=labels)
            derivative_loss = self.mse_derivative_loss(prediction=prediction, ground_truth=labels)
            self.backward(derivative_loss)
            self.update()

            accuracy = np.mean(np.round(prediction) == labels)
            if epoch % 1 == 0:
                print(f'Epoch = {epoch}  Loss = [{loss:.8f}]  Accuracy = {accuracy:.2f}')
                self.learning_epoch.append(epoch)
                self.learning_loss.append(loss)

            if loss < 0.001:
                break
                
    def predict(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Predict the labels of inputs
        :param inputs: input data
        :return: predict labels
        """

        prediction = self.forward(inputs=inputs)
        #print(prediction)
        predictions_rounded =  np.round(prediction)

        # Print details for each prediction
        for idx, (pred, label) in enumerate(zip(prediction, labels)):
            print(f"Iter: {idx} \t| Ground Truth: {label} | prediction: {pred} \t|")
        
        return predictions_rounded
    def show_result(self, inputs: np.ndarray, labels: np.ndarray, data_type: int) -> None:
        # Determine data type description
        if data_type == 0:
            data_type_desc = "Linear"
        else:
            data_type_desc = "XOR"
        # Plot ground truth and prediction
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f'Ground truth - {data_type_desc}', fontsize=18)
        for idx, point in enumerate(inputs):
            if labels[idx][0] == 0:
                plt.plot(point[0], point[1], 'ro')  
            else:
                plt.plot(point[0], point[1], 'bo')  

        pred_labels = self.predict(inputs,labels)
        plt.subplot(1, 2, 2)
        plt.title(f'Predict result - {data_type_desc} Acc:[{float(np.sum(pred_labels == labels)) / len(labels):.2f}]', fontsize=18)
        for idx, point in enumerate(inputs):
            if pred_labels[idx][0] == 0:
                plt.plot(point[0], point[1], 'ro')  
            else:
                plt.plot(point[0], point[1], 'bo') 

        print('-------Experiment Details -------\n')
        print(f'Data type: {data_type_desc} data points')
        print(f'Activation: {self.activation}')
        print(f'Learning rate: {self.learning_rate}')
        print(f'Number of hidden layers: {len(self.hidden_layers)}')
        print(f'Hidden units: {self.hidden_units}')
        print(f'Optimizer: {self.optimizer}')
        print(f'Accuracy: {float(100*np.sum(pred_labels == labels)) / len(labels):.2f}%')
        print(f'Loss: {np.mean(self.learning_loss):.4f}')
        print('-------------------------')


        # Plot learning curve
        plt.figure(figsize=(10, 5))
        plt.title('Learning curve', fontsize=18)
        plt.plot(self.learning_epoch, self.learning_loss, label="Training Loss", color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def mse_loss(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        return np.mean((prediction - ground_truth) ** 2)

    @staticmethod
    def mse_derivative_loss(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        return 2 * (prediction - ground_truth) / len(ground_truth)

def main() -> None:
    print("Welcome to the Neural Network Configuration!")

    data_type = int(input("Choose the data type (0: Linear, 1: XOR): "))
    while data_type not in [0, 1]:
        print("Invalid choice. Please enter 0 for Linear or 1 for XOR.")
        data_type = int(input("Choose the data type (0: Linear, 1: XOR): "))

    number_of_data = int(input("Enter the number of data points (default 100): "))
    epoch = int(input("Enter the number of epochs (default 500000): "))
    learning_rate = float(input("Enter the learning rate (default 0.1): "))
    units = int(input("Enter the number of units in each hidden layer (default 4): "))
    
    num_of_hidden_layers = int(input("Enter the number of hidden layers (default 2): "))

    print("Choose the type of activation function:")
    print("1: Sigmoid")
    print("2: Tanh")
    print("3: ReLU")
    print("4: Leaky ReLU")
    print("5: None")
    activation_choice = int(input("Enter your choice: "))
    activation_dict = {1: 'sigmoid', 2: 'tanh', 3: 'relu', 4: 'leaky_relu', 5: 'none'}
    activation = activation_dict.get(activation_choice, 'none')  # default to 'none' if out of range

    print("Choose the type of optimizer:")
    print("1: Gradient Descent (gd)")
    print("2: Momentum")
    print("3: Adagrad")
    print("4: Adam")
    optimizer_choice = int(input("Enter your choice: "))
    optimizer_dict = {1: 'gd', 2: 'momentum', 3: 'adagrad', 4: 'adam'}
    optimizer = optimizer_dict.get(optimizer_choice, 'adam')

    # Generate data points
    if data_type == 0:
        inputs, labels = Generate_Data.linear(number_of_data)
    else:
        inputs, labels = Generate_Data.XOR_easy(number_of_data)

    neural_network = Model(epoch=epoch,
                           learning_rate=learning_rate,
                           num_of_hidden_layers=num_of_hidden_layers,
                           input_units=2,  # Default input units to 2 for XOR/Linear
                           hidden_units=units,
                           output_units=1,  # Default output units to 1 for binary classification
                           activation=activation,
                           optimizer=optimizer)
    neural_network.train(inputs, labels)
    neural_network.show_result(inputs, labels, data_type)

if __name__ == '__main__':
    main()

