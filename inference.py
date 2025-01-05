import numpy as np
from tensorflow.keras.models import load_model
from config import SEQUENCE_LENGTH

class FallDetector:
    def __init__(self, model_path):
        """
        Initialize the FallDetector with a pre-trained LSTM model.

        Args:
            model_path (str): Path to the pre-trained LSTM model (.h5 file).
        """
        self.model = load_model(model_path)

    def predict(self, sequence):
        """
        Predict whether a fall has occurred based on a sequence of motion features.

        Args:
            sequence (numpy.ndarray): A 2D array containing a sequence of extracted motion and orientation features. 
                                      The first dimension corresponds to the sequence length, and the second dimension 
                                      corresponds to feature dimensions.

        Returns:
            tuple: 
                - bool: Whether a fall is predicted (True/False).
                - float: The probability score of the fall prediction.
        """
        if len(sequence) < SEQUENCE_LENGTH:
            return False, 0.0

        lstm_input = np.array([sequence])
        prediction = self.model.predict(lstm_input)
        probability = float(prediction[0][0])
        is_fall = probability <= 0.5
        return is_fall, probability
