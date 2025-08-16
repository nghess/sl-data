# Neural Population Classifiers Guide

## Overview

This guide covers the binary classification pipeline for predicting behavioral states from neural population activity using the SessionData class. We implemented both MLP and LSTM approaches for different temporal modeling scenarios.

## Files Created

### 1. `neural_classifier.py`
Core module containing all model classes and training utilities.

### 2. `classification_demo.py` 
Demo script for MLP-based classification using individual time points.

### 3. `lstm_demo.py`
Demo script for LSTM-based classification using temporal sequences.

---

## Models

### MLP Classifier (`PopulationMLP`)
- **Purpose**: Predict behavioral state from single population vectors
- **Input**: `[batch_size x n_neurons]`
- **Architecture**: Configurable hidden layers → ReLU → Dropout → Sigmoid
- **Default**: 256 → 128 → 64 → 1 neurons
- **Use Case**: When temporal context isn't critical

### LSTM Classifier (`PopulationLSTM`)
- **Purpose**: Predict behavioral state from population sequences
- **Input**: `[batch_size x sequence_length x n_neurons]`
- **Architecture**: LSTM → Hidden state → MLP classifier
- **Default**: 2-layer LSTM (64 hidden) → 32 → 1 neurons
- **Use Case**: When temporal dynamics are important

---

## Data Preparation

### For MLP (Individual Time Points)
```python
# Create population matrix [time_bins x neurons]
pop_matrix, time_bins, cluster_ids = session.create_population_raster(
    bin_size_ms=100.0,
    zscore_neurons=True
)
population_matrix = pop_matrix.T

# Get behavioral labels for each time bin
labels = session.create_event_colormap(
    time_bins, 
    event_column='flip_state', 
    aggregation='any'
)
```

### For LSTM (Sequences)
```python
# First create population matrix as above, then:
sequences, sequence_labels = create_sequences(
    population_matrix, 
    labels, 
    sequence_length=10,  # 10 time bins per sequence
    stride=5            # 5 bin stride (50% overlap)
)
# Results in [n_sequences x sequence_length x n_neurons]
```

---

## Usage Examples

### MLP Classification
```python
from classification_demo import run_classification_demo

classifier, metrics, history = run_classification_demo(
    mouse_id="7003",
    session_id="m4", 
    experiment="clickbait-motivate",
    event_column="click",
    bin_size_ms=100.0,
    epochs=1000
)
```

### LSTM Classification
```python
from lstm_demo import run_lstm_demo

classifier, metrics, history = run_lstm_demo(
    mouse_id="7003",
    session_id="m4", 
    experiment="clickbait-motivate",
    event_column="drinking",
    sequence_length=10,
    bin_size_ms=100.0,
    stride=5,
    epochs=500
)
```

---

## Key Parameters

### Common Parameters
- **`event_column`**: Behavioral variable to predict (e.g., 'flip_state', 'click', 'drinking')
- **`bin_size_ms`**: Time bin size in milliseconds
- **`epochs`**: Number of training epochs
- **`test_size`**: Fraction of data for testing (default: 0.2)

### MLP-Specific
- **`hidden_sizes`**: List of hidden layer sizes (default: [256, 128, 64])

### LSTM-Specific
- **`sequence_length`**: Number of time bins per sequence (default: 10)
- **`stride`**: Step size between sequences (default: 5)
- **`hidden_size`**: LSTM hidden state size (default: 64)
- **`num_layers`**: Number of LSTM layers (default: 2)
- **`bidirectional`**: Use bidirectional LSTM (default: False)

---

## Performance Metrics

Both models provide:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

---

## CUDA Setup

**Issue**: PyTorch may be installed as CPU-only version
**Solution**: Reinstall with CUDA support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verification**:
```python
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device name: {torch.cuda.get_device_name(0)}')
```

---

## Architecture Details

### MLP Wrapper (`PopulationClassifier`)
- Handles training loop with validation monitoring
- Automatic device detection (CUDA/CPU)
- Built-in evaluation metrics
- Easy prediction interface

### LSTM Wrapper (`SequenceClassifier`)
- Same interface as MLP classifier
- Handles sequence data automatically
- Supports bidirectional processing
- Uses final hidden state for classification

### Datasets
- **`PopulationDataset`**: For MLP [n_samples x n_neurons]
- **`SequenceDataset`**: For LSTM [n_samples x sequence_length x n_neurons]

---

## Visualizations

### Training History
- Loss curves (training/validation)
- Accuracy progression
- Automatic plotting in demo scripts

### Sequence Examples (LSTM)
- Heatmaps of population activity sequences
- True vs. false label examples
- Prediction confidence visualization

---

## Next Steps / TODOs

1. **Model Comparison**: Compare MLP vs LSTM performance on same data
2. **Hyperparameter Tuning**: Grid search for optimal architectures
3. **Cross-Validation**: Implement k-fold validation
4. **Feature Analysis**: Investigate which neurons/patterns are most predictive
5. **Temporal Analysis**: Vary sequence lengths and strides
6. **Multi-Class**: Extend to predict multiple behavioral states
7. **Regularization**: Add L1/L2 regularization options
8. **Early Stopping**: Implement early stopping based on validation loss

---

## Troubleshooting

### Common Issues
1. **Class Imbalance**: Check positive class proportion, consider resampling
2. **Memory Issues**: Reduce batch size or sequence length
3. **Overfitting**: Increase dropout rate or reduce model complexity
4. **Poor Performance**: Check data quality, try different bin sizes
5. **CUDA OOM**: Reduce batch size or model size

### Data Quality Checks
- Verify event column exists in SessionData.events
- Check for reasonable class balance (not <1% or >99%)
- Ensure sufficient data after sequence creation
- Validate time alignment between neural and behavioral data