import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import Utility

np.random.seed(42)

class NeuralNet:
    def __init__(self, hidden_layer_sizes=(4,), activation='identity', batch_size=1,
                 learning_rate=0.1, eta_decay=False, warm_start=False, 
                 early_stopping=False, patience=0, epoch=200):
      self.n_layers  = len(hidden_layer_sizes)
      self.n_inputs  = 0
      self.n_outputs = 0
      self.n_epoch   = epoch
      self.batch_size  = batch_size
      self.early_stopping = early_stopping
      self.patience       = patience
      self.learning_rate = learning_rate
      self.eta_decay     = eta_decay
      self.warm_start    = warm_start
      self.has_trained   = False
      
      self.weights = [None] * (self.n_layers + 1) # +1: output layer
      self.biases  = [None] * (self.n_layers + 1)
      self.Z       = [None] * (self.n_layers + 1)
      self.A       = [None] * (self.n_layers + 1)
      self.df      = [None] * (self.n_layers + 1)
      
      self.hidden_layer_sizes = hidden_layer_sizes # To later determine
                                                    # weights matrices dim.
      
      if   activation == 'tanh'    : self.activation = Utility.tanh
      elif activation == 'sigmoid' : self.activation = Utility.sigmoid
      elif activation == 'relu'    : self.activation = Utility.relu
      elif activation == 'sintr'   : self.activation = Utility.sintr 
      else : self.activation = Utility.identity
          
      return None
    
    def __weights_initialization(self, X, y):
      # Initialize hidden layers
      n_cols  = X.shape[1] # nb of features of X
      
      self.n_inputs = X.shape[1]
      for l in range (0, self.n_layers):
          n_lines         = self.hidden_layer_sizes[l]
          self.weights[l] = np.random.uniform(-1, 1, (n_lines,n_cols))
          self.biases[l]  = np.random.uniform(-1, 1, (n_lines, 1)) # bias: vector
          n_cols          = n_lines
      
      # Initialize output layer
      l_out = self.n_layers # index of the last layer
      self.n_outputs = y.shape[1]
      self.weights[l_out] = np.random.uniform(-1, 1, (self.n_outputs,n_cols) )
      self.biases[l_out]  = np.random.uniform(-1, 1, (self.n_outputs, 1) )
      
      return None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, val=0.2):
      if not self.has_trained:
        self.__weights_initialization(X_train, y_train)
      
      if X_val is None:
        X_train, X_val, y_train, y_val = \
          train_test_split(X_train, y_train, test_size=val, random_state=42)
      
      epoch_train_error = []
      epoch_val_error  = []
      
      # Create progress bar for epochs
      epoch_pbar = tqdm(range(self.n_epoch), desc="Training Progress")
      
      for e in epoch_pbar:
          ave_train_error = 0
          batch_train_error = []
          processed_batch = 0
          
          # Re-sample the training data
          X_train, y_train = shuffle(X_train, y_train) # sklearn
          
          # Calculate number of batches
          n_batches = X_train.shape[0] // self.batch_size
          if X_train.shape[0] % self.batch_size != 0:
              n_batches += 1
          
          # Create progress bar for batches
          batch_pbar = tqdm(range(0, X_train.shape[0], self.batch_size), 
                          desc=f"Epoch {e+1}/{self.n_epoch}", 
                          leave=False)
          
          # Training / Go over the data in batches
          for start in batch_pbar:
              batch_X_train = X_train[start:start+self.batch_size].transpose()
              batch_y_train = y_train[start:start+self.batch_size].transpose()
              
              # Forward pass
              batch_train_error.append(self.__feed_forward(batch_X_train, batch_y_train)[0])
              ave_train_error += batch_train_error[processed_batch]
              
              # Backpropagation
              self.__backward_pass(batch_X_train, batch_y_train, e)
              
              processed_batch += 1
              
              # Update batch progress bar
              batch_pbar.set_postfix({
                  'Train Error': f"{batch_train_error[-1]:.4f}",
                  'Batch': f"{processed_batch}/{n_batches}"
              })
          
          # Close batch progress bar
          batch_pbar.close()
          
          # Training error
          ave_train_error /= len(batch_train_error)
          epoch_train_error.append(ave_train_error)
              
          # Validation error
          val_error = self.__feed_forward(X_val.transpose(), y_val.transpose())[0]
          epoch_val_error.append(val_error)
          
          # Update epoch progress bar
          epoch_pbar.set_postfix({
              'Train Error': f"{ave_train_error:.4f}",
              'Val Error': f"{val_error:.4f}"
          })
          
          if e % 10 == 0:
              print("\n* Epoch " + str(e) + " -- Error :   Train : " + \
                    "{:.4f}".format(ave_train_error) + "    Validation : "+ \
                        "{:.4f}".format(val_error))
          
          #### END OF CURRENT EPOCH ####
      #### END OF EPOCHS ####
      
      self.has_trained = True
      
      # Plot error as a function of training epoch
      fig = plt.figure(figsize=(12,8))
      fig.suptitle('Evolution of error during training', fontsize=20)
      plt.plot(epoch_train_error, label="Train")
      plt.plot(epoch_val_error,  label="Validation")
      plt.legend()
      plt.xlabel('Epoch of training', fontsize=20)
      plt.ylabel('Error', fontsize=20)
      plt.show()
      
      return None
    
    def predict(self, X_batch):
      """
      Compute the output for instances in X_batch
      Returns: output probabilities
      """
      _, probabilities = self.__feed_forward(X_batch.transpose())
      return probabilities.transpose()
    
    def __feed_forward(self, X_batch, y_batch=None):
      """
      Performs a forward pass using batch X_batch
        - Feed the batch through each hidden layer, and compute the output vector
        - Compute the error between output and ground truth y_batch using cross entropy loss
      Parameters:
        X_batch: batch used in forward pass
        y_batch: labels for X_batch
      Returns:
        model error on batch, output probabilities
      """
      A = X_batch
      
      # Feed input signal through the hidden layers
      for l in range(self.n_layers):
          # Linear transformation
          Z = np.dot(self.weights[l], A) + self.biases[l]
          # Activation
          A, self.df[l] = self.activation(Z)
          # Store activations and outputs for backprop
          self.Z[l] = Z
          self.A[l] = A
      
      # Compute the output (last layer)
      l_out = self.n_layers
      Z = np.dot(self.weights[l_out], A) + self.biases[l_out]
      predictions = Utility.softmax(Z)
      self.Z[l_out] = Z
      self.A[l_out] = predictions
      
      # Compute the error if y_batch is provided
      error = 0
      if y_batch is not None:
          error = Utility.cross_entropy_cost(predictions, y_batch)
      
      return error, predictions
    
    def __backward_pass(self, X_batch, y_batch, epoch):
      """
      Perform gradient backpropagation
        (ASSUMES output softmax activation & cross-entropy cost)
        About biases update for batch size > 1:
        https://stats.stackexchange.com/questions/373163/how-are-biases-updated-when-batch-size-1
      Parameters:
        X_batch : batch used in forward pass
        y_batch : labels for X_batch
        epoch   : epoch # 
      Returns : None
      """
      delta = [None] * (self.n_layers + 1)
      dW = [None] * (self.n_layers + 1)
      db = [None] * (self.n_layers + 1)
      
      # Error on output layer
      l_out = self.n_layers
      delta[l_out] = self.A[l_out] - y_batch
      
      # Backpropagate the error in the hidden layers
      for l in range(l_out-1, -1, -1):
          # Compute delta for layer l
          delta[l] = np.dot(self.weights[l+1].T, delta[l+1]) * self.df[l]
      
      # Update the parameters
      for l in range(l_out, -1, -1):
          # Compute gradients
          if l == 0:
              dW[l] = np.dot(delta[l], X_batch.T) / self.batch_size
          else:
              dW[l] = np.dot(delta[l], self.A[l-1].T) / self.batch_size
          db[l] = self.__ave_delta(delta[l])
          
          # Update weights and biases
          if self.eta_decay:
              eta = self.learning_rate / (1 + epoch/10)
          else:
              eta = self.learning_rate
              
          self.weights[l] -= eta * dW[l]
          self.biases[l] -= eta * db[l]
      
      return None
    
    @staticmethod
    def __ave_delta(delta):
      return np.array([delta.mean(axis=1)]).transpose()
    
    def __str__(self):
      output = ""
      for l in range(0, self.n_layers+1): # +1: output layer
          output += "---- LAYER " + str(l) + "\n"
          output += "  * Weights"+ str(self.weights[l].shape) + "\n"
          output += str(self.weights[l]) + "\n"
          output += "  * Biases"+ str(self.biases[l].shape) + "\n"
          output += str(self.biases[l]) + "\n"
          if self.A[l] is not None:
              output += "  * Activations"+ str(self.A[l].shape) + "\n"
              output += str(self.A[l]) + "\n"
      return output
    
    def __repr__(self):
      return self.__str__()

    def save_weights(self, filename):
        """Save the weights and biases of the network to a file."""
        # Convert lists to arrays with dtype=object to handle different sizes
        weights_array = np.array(self.weights, dtype=object)
        biases_array = np.array(self.biases, dtype=object)
        
        np.savez(filename, 
                 weights=weights_array,
                 biases=biases_array,
                 hidden_layer_sizes=self.hidden_layer_sizes,
                 n_inputs=self.n_inputs,
                 n_outputs=self.n_outputs)
        print(f"Model weights saved to {filename}")
    
    def load_weights(self, filename):
        """Load the weights and biases from a file."""
        data = np.load(filename, allow_pickle=True)
        
        # Load network architecture
        self.hidden_layer_sizes = tuple(data['hidden_layer_sizes'])
        self.n_inputs = int(data['n_inputs'])
        self.n_outputs = int(data['n_outputs'])
        self.n_layers = len(self.hidden_layer_sizes)
        
        # Load weights and biases
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        
        # Initialize other arrays
        self.Z = [None] * (self.n_layers + 1)
        self.A = [None] * (self.n_layers + 1)
        self.df = [None] * (self.n_layers + 1)
        
        self.has_trained = True
        print(f"Model weights loaded from {filename}")


###############################################################################
# MODULE TEST
###############################################################################
import pandas as pd

def main(): 
    # Load data
    root = ""
    iris_df = pd.read_csv(root + 'iris.csv')
    
    # Extrate attributes names and target
    df_columns = iris_df.columns.values.tolist()
    features = df_columns[0:4]
    label    = df_columns[4:] # ['class']
    
    X = iris_df[features]
    y = iris_df[label]
    y = pd.get_dummies(y, dtype='int') # one-hot encoding
    
    X_train, X_val, y_train, y_val = \
      train_test_split(X, y, test_size=0.2, random_state=42)
    
    # df to numpy arrays
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    
    nn = NeuralNet(hidden_layer_sizes=(16,8), batch_size=4, activation='tanh',
                   learning_rate=0.01, epoch=100)
    
    nn.fit(X_train, y_train, X_val, y_val)
    
    return

if __name__ == '__main__':
    main()