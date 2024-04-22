from bw_to_color_model.colorizer_model import ECCVGenerator


def train_model(X_train, y_train, epochs=10, batch_size=32):
    model = ECCVGenerator()
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
