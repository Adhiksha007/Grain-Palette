import numpy as np
import matplotlib.pyplot as plt

# Predict on new image - create a function
def predict_rice_type(model, img_array, class_names):
  prediction = model.predict(img_array)
  predicted_index = np.argmax(prediction[0])  # 🔥 gets index of highest probability
  predicted_class = class_names[predicted_index]
  confidence = prediction[0][predicted_index] * 100
  return predicted_class, confidence
