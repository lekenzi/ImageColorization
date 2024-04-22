import tensorflow as tf
import os

def load_images_from_dir(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(directory, filename))
    
    images = []
    for image_path in image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [128, 128])  # Adjust the size to 128x128
        images.append(image)
    
    return images


def preprocess_images(x_images, y_images, batch_size):
    # Combine x_images and y_images into a single dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_images, y_images))
    
    # Preprocess each image in the dataset
    def preprocess_image(x_image, y_image):
        x_image = tf.io.read_file(x_image)
        x_image = tf.image.decode_image(x_image, channels=3)
        x_image = tf.image.resize(x_image, [128, 128])
        
        y_image = tf.io.read_file(y_image)
        y_image = tf.image.decode_image(y_image, channels=3)
        y_image = tf.image.resize(y_image, [128, 128])
        
        return x_image, y_image
    
    dataset = dataset.map(preprocess_image)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset



