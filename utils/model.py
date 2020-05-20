"""Network to classify a picture between the class polo or t-shirt"""
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

net_model = models.Sequential()
net_model.add(layers.Conv2D(6, (7, 7), activation='relu', input_shape=(80, 80, 3), kernel_regularizer=regularizers.l2(0.001)))
net_model.add(layers.MaxPooling2D((2, 2)))
net_model.add(layers.Conv2D(16, (6, 6), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
net_model.add(layers.MaxPooling2D((4, 4)))
net_model.add(layers.Conv2D(16, (8, 8), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
net_model.add(layers.Flatten())
net_model.add(layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
net_model.add(layers.dropout(0.5))
net_model.add(layers.Dense(6, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
net_model.add(layers.dropout(0.5))
net_model.add(layers.Dense(2))
