import pickle
import numpy as np

# Load model
model = pickle.load(open('../models/model.pkl', 'rb'))

def predict_delivery_time(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]


# Example test
if __name__ == "__main__":
    sample = [10, 20, 2, 1, 0, 1, 0, 1, 0, 0]  # must match training features
    print("Predicted Time:", predict_delivery_time(sample))