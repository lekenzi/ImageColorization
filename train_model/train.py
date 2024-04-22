from bw_to_color_model.colorizer_model import ECCVGenerator


def train_model(dataset, epochs=10, batch_size=32):
    model = ECCVGenerator()
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    history = model.fit(dataset, epochs=epochs, batch_size=batch_size)
    
    # Save the model
    model.save('/path/to/save/model.h5')
    
    # Save the history object
    with open('/history/history.obj', 'wb') as file:
        pickle.dump(history.history, file)
