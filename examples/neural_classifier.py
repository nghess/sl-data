"""
Simple MLP binary classifier for neural population data.

This module provides a PyTorch-based MLP for predicting behavioral states
from neural population activity vectors.

Code by Nate Gonzales-Hess, August 2025.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class PopulationDataset(Dataset):
    """Dataset class for population vectors and behavioral labels."""
    
    def __init__(self, population_vectors: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        population_vectors : np.ndarray
            Population activity matrix [n_samples x n_neurons]
        labels : np.ndarray
            Binary labels [n_samples]
        """
        self.population_vectors = torch.FloatTensor(population_vectors)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.population_vectors)
    
    def __getitem__(self, idx):
        return self.population_vectors[idx], self.labels[idx]


class PopulationMLP(nn.Module):
    """Simple MLP for binary classification of population activity."""
    
    def __init__(self, input_size: int, hidden_sizes: list = [256, 128, 64], 
                 dropout_rate: float = 0.2):
        """
        Initialize MLP classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (number of neurons)
        hidden_sizes : list
            List of hidden layer sizes
        dropout_rate : float
            Dropout probability for regularization
        """
        super(PopulationMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)  # Only squeeze the last dimension, preserve batch dim


class PopulationClassifier:
    """Wrapper class for training and evaluating population classifiers."""
    
    def __init__(self, input_size: int, hidden_sizes: list = [128, 64], 
                 dropout_rate: float = 0.2, device: str = 'cuda'):
        """
        Initialize classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (number of neurons)
        hidden_sizes : list
            List of hidden layer sizes
        dropout_rate : float
            Dropout probability for regularization
        device : str
            Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = PopulationMLP(input_size, hidden_sizes, dropout_rate).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = None
        
    def train_model(self, train_data: PopulationDataset, val_data: Optional[PopulationDataset] = None,
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                   verbose: bool = True, use_best_model: bool = False) -> dict:
        """
        Train the classifier.
        
        Parameters:
        -----------
        train_data : PopulationDataset
            Training dataset
        val_data : PopulationDataset, optional
            Validation dataset
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        history : dict
            Training history with losses and metrics
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size) if val_data else None
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # Best model tracking
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_true = []
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        val_preds.extend((outputs > 0.5).cpu().numpy())
                        val_true.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_acc = accuracy_score(val_true, val_preds)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Restore best model if validation was used and use_best_model is True
        if best_model_state is not None and use_best_model:
            self.model.load_state_dict(best_model_state)
            # Find the epoch with best validation loss
            best_epoch = history['val_loss'].index(best_val_loss)
            history['best_epoch'] = best_epoch
            if verbose:
                print(f"Restored best model from epoch {best_epoch + 1} with val_loss = {best_val_loss:.4f}")
        elif best_model_state is not None:
            # Still track best epoch for plotting, but don't restore
            best_epoch = history['val_loss'].index(best_val_loss)
            history['best_epoch'] = best_epoch
            if verbose:
                print(f"Using final model (epoch {epochs}). Best val_loss was at epoch {best_epoch + 1} = {best_val_loss:.4f}")
        
        return history
    
    def evaluate(self, test_data: PopulationDataset, batch_size: int = 32) -> dict:
        """
        Evaluate the classifier on test data.
        
        Parameters:
        -----------
        test_data : PopulationDataset
            Test dataset
        batch_size : int
            Batch size for evaluation
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=batch_size)
        
        all_preds = []
        all_probs = []
        all_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                predictions = (outputs > 0.5).cpu().numpy()
                probabilities = outputs.cpu().numpy()
                
                all_preds.extend(predictions)
                all_probs.extend(probabilities)
                all_true.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_true, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='binary')
        auc = roc_auc_score(all_true, all_probs)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics, all_true, all_preds, all_probs
    
    def predict(self, population_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        population_vectors : np.ndarray
            Population activity matrix [n_samples x n_neurons]
            
        Returns:
        --------
        predictions : np.ndarray
            Binary predictions
        probabilities : np.ndarray
            Prediction probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(population_vectors).to(self.device)
            outputs = self.model(x)
            predictions = (outputs > 0.5).cpu().numpy()
            probabilities = outputs.cpu().numpy()
        
        return predictions, probabilities


class SequenceDataset(Dataset):
    """Dataset class for population vector sequences and behavioral labels."""
    
    def __init__(self, population_sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize sequence dataset.
        
        Parameters:
        -----------
        population_sequences : np.ndarray
            Population activity sequences [n_samples x sequence_length x n_neurons]
        labels : np.ndarray
            Binary labels [n_samples]
        """
        self.population_sequences = torch.FloatTensor(population_sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.population_sequences)
    
    def __getitem__(self, idx):
        return self.population_sequences[idx], self.labels[idx]


class PopulationLSTM(nn.Module):
    """Simple LSTM for binary classification of population activity sequences."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout_rate: float = 0.2, bidirectional: bool = False):
        """
        Initialize LSTM classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (number of neurons)
        hidden_size : int
            LSTM hidden state size
        num_layers : int
            Number of LSTM layers
        dropout_rate : float
            Dropout probability for regularization
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(PopulationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        # If bidirectional, hidden is [num_layers*2, batch, hidden_size]
        # Otherwise, hidden is [num_layers, batch, hidden_size]
        if self.bidirectional:
            # Concatenate final forward and backward hidden states
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # Use final hidden state
            final_hidden = hidden[-1]
        
        # Classify based on final hidden state
        output = self.classifier(final_hidden)
        
        return output.squeeze(-1)  # Only squeeze the last dimension, preserve batch dim


class SequenceClassifier:
    """Wrapper class for training and evaluating sequence classifiers."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout_rate: float = 0.2, bidirectional: bool = False, device: str = 'cuda'):
        """
        Initialize sequence classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (number of neurons)
        hidden_size : int
            LSTM hidden state size
        num_layers : int
            Number of LSTM layers
        dropout_rate : float
            Dropout probability for regularization
        bidirectional : bool
            Whether to use bidirectional LSTM
        device : str
            Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = PopulationLSTM(
            input_size, hidden_size, num_layers, dropout_rate, bidirectional
        ).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = None
        
    def train_model(self, train_data: SequenceDataset, val_data: Optional[SequenceDataset] = None,
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                   verbose: bool = True, use_best_model: bool = False) -> dict:
        """
        Train the sequence classifier.
        
        Parameters:
        -----------
        train_data : SequenceDataset
            Training dataset
        val_data : SequenceDataset, optional
            Validation dataset
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        history : dict
            Training history with losses and metrics
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size) if val_data else None
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # Best model tracking
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_true = []
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        val_preds.extend((outputs > 0.5).cpu().numpy())
                        val_true.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_acc = accuracy_score(val_true, val_preds)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Restore best model if validation was used and use_best_model is True
        if best_model_state is not None and use_best_model:
            self.model.load_state_dict(best_model_state)
            # Find the epoch with best validation loss
            best_epoch = history['val_loss'].index(best_val_loss)
            history['best_epoch'] = best_epoch
            if verbose:
                print(f"Restored best model from epoch {best_epoch + 1} with val_loss = {best_val_loss:.4f}")
        elif best_model_state is not None:
            # Still track best epoch for plotting, but don't restore
            best_epoch = history['val_loss'].index(best_val_loss)
            history['best_epoch'] = best_epoch
            if verbose:
                print(f"Using final model (epoch {epochs}). Best val_loss was at epoch {best_epoch + 1} = {best_val_loss:.4f}")
        
        return history
    
    def evaluate(self, test_data: SequenceDataset, batch_size: int = 32) -> dict:
        """
        Evaluate the sequence classifier on test data.
        
        Parameters:
        -----------
        test_data : SequenceDataset
            Test dataset
        batch_size : int
            Batch size for evaluation
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=batch_size)
        
        all_preds = []
        all_probs = []
        all_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                predictions = (outputs > 0.5).cpu().numpy()
                probabilities = outputs.cpu().numpy()
                
                all_preds.extend(predictions)
                all_probs.extend(probabilities)
                all_true.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_true, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='binary')
        auc = roc_auc_score(all_true, all_probs)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics, all_true, all_preds, all_probs
    
    def predict(self, population_sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new sequence data.
        
        Parameters:
        -----------
        population_sequences : np.ndarray
            Population activity sequences [n_samples x sequence_length x n_neurons]
            
        Returns:
        --------
        predictions : np.ndarray
            Binary predictions
        probabilities : np.ndarray
            Prediction probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(population_sequences).to(self.device)
            outputs = self.model(x)
            predictions = (outputs > 0.5).cpu().numpy()
            probabilities = outputs.cpu().numpy()
        
        return predictions, probabilities