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
        image = tf.image.resize(image, [224, 224])  # Adjust the size as needed
        image = image / 255.0  # Normalize the pixel values
        images.append(image)
    
    return images


def preprocess_images(images, batch_size):
    # Convert images to black and white
    bw_images = []
    for image in images:
        bw_image = tf.image.rgb_to_grayscale(image)
        bw_images.append(bw_image)
    
    # Create a dataset from the black and white images
    dataset = tf.data.Dataset.from_tensor_slices(bw_images)
    
    # Prefetch the dataset
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset



