from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Bidirectional

def build_model(input_short_shape, input_long_shape):
    input_short = Input(shape=input_short_shape)
    input_long = Input(shape=input_long_shape)
    x1 = Bidirectional(LSTM(64))(input_short)
    x2 = Bidirectional(LSTM(64))(input_long)
    x = concatenate([x1, x2])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_short, input_long], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
