from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense


def create_model():
	model = keras.Sequential()
	model.add(Dropout(rate=0.2, input_shape=(9270, )))
	model.add(Dense(units=64, activation='relu'))
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=64, activation='relu'))
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=1, activation='sigmoid'))

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

	return model
