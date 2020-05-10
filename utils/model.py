from tensorflow.keras import layers, models

net_model = models.Sequential()
net_model.add(layers.Conv2D(6, (7, 7), activation='relu', input_shape=(224, 224, 3)))
net_model.add(layers.MaxPooling2D((2, 2)))
net_model.add(layers.Conv2D(16, (6, 6), activation='relu'))
net_model.add(layers.MaxPooling2D((4, 4)))
net_model.add(layers.Conv2D(16, (8, 8), activation='relu'))
net_model.add(layers.Flatten())
net_model.add(layers.Dense(20, activation='relu'))
net_model.add(layers.Dense(6, activation='relu'))
net_model.add(layers.Dense(3))

