import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors
import pandas as pd
import threading
import tkinter.font as tkFont

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.weights.append(np.random.randn(prev_size, hidden_size) * np.sqrt(2 / prev_size))
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size

        self.weights.append(np.random.randn(prev_size, output_size) * np.sqrt(2 / prev_size))
        self.biases.append(np.zeros((1, output_size)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability fix for large values
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def categorical_crossentropy(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.layer_outputs[-1], w) + b
            self.layer_inputs.append(z)

            if i < len(self.weights) - 1:  # Hidden layers
                if self.activation_name == 'sigmoid':
                    self.layer_outputs.append(self.sigmoid(z))
                elif self.activation_name == 'relu':
                    self.layer_outputs.append(self.relu(z))
            else:  # Output layer
                self.layer_outputs.append(self.softmax(z))
        print(f"Forward pass output shape: {self.layer_outputs[-1].shape}")
        return self.layer_outputs[-1]


    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        delta = self.layer_outputs[-1] - y
        for i in reversed(range(len(self.weights))):
            self.d_weights[i] = np.dot(self.layer_outputs[i].T, delta) / m
            self.d_biases[i] = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                if self.activation_name == 'sigmoid':
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.layer_inputs[i - 1])
                elif self.activation_name == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.layer_inputs[i - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.d_weights[i]
            self.biases[i] -= learning_rate * self.d_biases[i]

    def train(self, X, y, epochs=1000, learning_rate=0.1, callback=None):
        print(f"Starting training loop for {epochs} epochs.")
        losses = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}...")

            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = self.categorical_crossentropy(output, y)
            losses.append(loss)
            print(f"Loss at epoch {epoch+1}: {loss}")

            # Backward pass
            self.backward(X, y, learning_rate)

            # Callback for UI updates
            if callback:
                callback(epoch, loss, output)

        print("Training loop completed.")
        return losses




    def predict(self, X):
        outputs = self.forward(X)
        return np.argmax(outputs, axis=1)  # Return class indices for multi-class

class NetworkVisualizer:
    def __init__(self, canvas, nn, width=1200, height=800):
        self.canvas = canvas
        self.nn = nn
        self.width = width
        self.height = height
        self.node_positions = []
        self.lines = []

    def interpolate_color(self, weight, min_weight, max_weight):
        normalized_weight = (weight - min_weight) / (max_weight - min_weight + 1e-9)
        purple = colors.hex2color("#FF00FF")
        cyan = colors.hex2color("#00FFFF")
        interpolated_color = [
            purple[i] + (cyan[i] - purple[i]) * normalized_weight
            for i in range(3)
        ]
        return colors.rgb2hex(interpolated_color)

    def draw_network(self, width=None, height=None):
        if width:
            self.width = width
        if height:
            self.height = height

        self.canvas.delete("all")
        self.node_positions = []
        self.lines = []

        layer_sizes = [self.nn.input_size] + self.nn.hidden_layers + [self.nn.output_size]
        layer_gap = self.width / (len(layer_sizes) + 1)

        # Centering calculation
        x_offset = (self.width - layer_gap * len(layer_sizes)) / 2
        y_offset = self.height / 2

        for i, size in enumerate(layer_sizes):
            x = x_offset + (i + 1) * layer_gap
            y_gap = self.height / (size + 1)
            layer_positions = [(x, y_offset + (j + 1) * y_gap - self.height / 2) for j in range(size)]
            self.node_positions.append(layer_positions)

        # Draw connections
        for i, layer in enumerate(self.node_positions[:-1]):
            next_layer = self.node_positions[i + 1]
            for j, pos in enumerate(layer):
                for k, next_pos in enumerate(next_layer):
                    line = self.canvas.create_line(
                        pos[0], pos[1], next_pos[0], next_pos[1],
                        fill="#444444", width=2
                    )
                    self.lines.append((line, j, k, i))

        # Draw neurons
        for layer in self.node_positions:
            for pos in layer:
                self.canvas.create_oval(
                    pos[0] - 10, pos[1] - 10, pos[0] + 10, pos[1] + 10,
                    fill="orange", outline="black"
                )

    def update_weights(self):
        min_weight = min(w.min() for w in self.nn.weights)
        max_weight = max(w.max() for w in self.nn.weights)

        for line, j, k, i in self.lines:
            weight = self.nn.weights[i][j, k]
            color = self.interpolate_color(weight, min_weight, max_weight)
            self.canvas.itemconfig(line, fill=color, width=min(max(abs(weight) * 5, 1), 5))

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network Visualization")
        self.master.attributes("-fullscreen", True)

        # Left frame for visualization
        self.left_frame = tk.Frame(self.master, bg="black", width=1200, height=800)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.left_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right frame for controls
        self.right_frame = tk.Frame(self.master, bg="white", width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Define a custom font for all buttons and labels
        custom_font = tkFont.Font(family="Helvetica", size=12)

        # Buttons, labels, and other widgets
        self.load_button = tk.Button(
            self.right_frame, text="Load Dataset", font=custom_font, command=self.load_dataset
        )
        self.load_button.pack(pady=10)

        self.hidden_layers_label = tk.Label(
            self.right_frame, text="Hidden Layers (comma-separated):", font=custom_font
        )
        self.hidden_layers_label.pack(pady=5)

        self.hidden_layers_entry = tk.Entry(self.right_frame, font=custom_font)
        self.hidden_layers_entry.insert(0, "6,8,6")
        self.hidden_layers_entry.pack(pady=5)

        self.activation_menu_label = tk.Label(
            self.right_frame, text="Activation Function:", font=custom_font
        )
        self.activation_menu_label.pack(pady=5)

        self.activation_var = tk.StringVar(value="sigmoid")
        self.activation_menu = tk.OptionMenu(
            self.right_frame, self.activation_var, "sigmoid", "relu", "tanh"
        )
        self.activation_menu.config(font=custom_font)
        self.activation_menu.pack(pady=5)

        self.train_button = tk.Button(
            self.right_frame, text="Train Neural Network", font=custom_font, command=self.train_network
        )
        self.train_button.pack(pady=10)

        self.accuracy_label = tk.Label(
            self.right_frame, text="Accuracy: N/A", font=custom_font
        )
        self.accuracy_label.pack(pady=10)

        self.quit_button = tk.Button(
            self.right_frame, text="Quit", font=custom_font, command=self.master.destroy
        )
        self.quit_button.pack(pady=10)

        # Add a plot to the right frame
        self.figure, (self.ax_loss, self.ax_accuracy) = plt.subplots(2, 1, figsize=(4, 6))
        self.figure.tight_layout(pad=3.0)
        self.ax_loss.set_title("Loss")
        self.ax_accuracy.set_title("Accuracy")
        self.loss_line, = self.ax_loss.plot([], [], label="Loss")
        self.accuracy_line, = self.ax_accuracy.plot([], [], label="Accuracy")
        self.ax_loss.legend()
        self.ax_accuracy.legend()
        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Neural Network and Visualization Setup
        self.nn = None
        self.visualizer = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.epochs = 100
        self.losses = []
        self.accuracies = []

        # Bind resize event
        self.canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        if self.visualizer:
            self.visualizer.draw_network(width=event.width, height=event.height)


    def update_plot(self, epoch, loss, predictions):
        self.losses.append(loss)
        if len(self.losses) > 1:
            x = list(range(len(self.losses)))
            self.loss_line.set_data(x, self.losses)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

        if self.y_train is not None:
            if self.y_train.shape[1] > 1:  # One-hot encoded
                predicted_classes = np.argmax(predictions, axis=1)
                actual_classes = np.argmax(self.y_train, axis=1)
                accuracy = np.mean(predicted_classes == actual_classes)
            else:  # Single-class labels
                accuracy = np.mean(np.round(predictions) == self.y_train.flatten())

            self.accuracies.append(accuracy)
            if len(self.accuracies) > 1:
                self.accuracy_line.set_data(x, self.accuracies)
                self.ax_accuracy.relim()
                self.ax_accuracy.autoscale_view()

        self.canvas_plot.draw()
        self.visualizer.update_weights()  # Update weights here


    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            # Load dataset
            data = pd.read_csv(file_path)

            # Drop 'Student_ID' column if it exists
            if 'Student_ID' in data.columns:
                data = data.drop(columns=['Student_ID'])

            # Separate features (X) and target (y)
            X = data.iloc[:, :-1].values  # All columns except the last
            y = data.iloc[:, -1].values  # Last column as target

            # Debug: Print initial shapes and unique target values
            print(f"Initial X shape: {X.shape}, Initial y shape: {y.shape}")
            print(f"Unique values in y before encoding: {np.unique(y)}")

            # Convert target values to integers if not already numeric
            stress_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
            y = np.array([stress_mapping[val] if isinstance(val, str) else val for val in y])

            # Check for single unique class
            if len(np.unique(y)) == 1:
                print("Only one unique class found in target variable. Keeping class indices.")
                y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
                self.y_train, self.y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
            else:
                # One-hot encode the target values (assuming multiple classes)
                y_one_hot = np.zeros((y.size, len(np.unique(y))))
                y_one_hot[np.arange(y.size), y] = 1

                # Debug: Print one-hot encoded shape
                print(f"One-hot encoded y shape: {y_one_hot.shape}")

                # Split into training and testing sets
                split_idx = int(0.8 * len(X))
                self.X_train, self.X_test = X[:split_idx], X[split_idx:]
                self.y_train, self.y_test = y_one_hot[:split_idx], y_one_hot[split_idx:]

            # Debug: Print final shapes
            print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

            # Normalize features
            self.X_train = (self.X_train - np.mean(self.X_train, axis=0)) / (np.std(self.X_train, axis=0) + 1e-9)
            self.X_test = (self.X_test - np.mean(self.X_test, axis=0)) / (np.std(self.X_test, axis=0) + 1e-9)

            messagebox.showinfo("Success", "Dataset loaded and preprocessed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def train_network(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return

        input_size = self.X_train.shape[1]
        output_size = self.y_train.shape[1]
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
        print(f"Input size: {input_size}, Output size: {output_size}")

        try:
            hidden_layers = list(map(int, self.hidden_layers_entry.get().split(',')))
            self.nn = NeuralNetwork(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, activation=self.activation_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize neural network: {e}")
            return

        self.visualizer = NetworkVisualizer(self.canvas, self.nn)
        self.visualizer.draw_network()

        def run_training():
            try:
                train_predictions = self.nn.predict(self.X_train)
                test_predictions = self.nn.predict(self.X_test)

                print("Training process started...")
                self.nn.train(
                    self.X_train,
                    self.y_train,
                    epochs=self.epochs,
                    learning_rate=0.1,
                    callback=self.update_plot
                )
                print("Training process completed.")

            except Exception as e:
                print(f"Error in training thread: {e}")
                messagebox.showerror("Error", f"Training failed: {e}")

        # Start the training thread
        training_thread = threading.Thread(target=run_training)
        training_thread.start()
        
        print("Training thread completed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
