from image_processor.image_preprocessor import load_images_from_dir, preprocess_images
from trainer.train import train_model

# Load images
directory = 'path_to_your_images'  # replace with your directory
images = load_images_from_dir(directory)

# Preprocess images
batch_size = 32  # replace with your batch size
preprocessed_images = preprocess_images(images, batch_size)

# Assuming y_train is defined somewhere above

# Train the model
train_model(preprocessed_images, y_train)