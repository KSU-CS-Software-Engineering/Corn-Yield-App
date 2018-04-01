from corn_app import csv_features
import numpy as np
import tensorflow as tf
import json
import os
import csv


MODELS_DIR = 'models'
ITERATIONS = 1000
N          = 2   # Number of features.

#expects float as param
def get_count(front_count, ratio):
    """
    Uses last trained model to predict the full kernel count

    Args:
        front_count(int): This is the count determined by the watershed/otsu method in the contours module
        ratio(float)    : 

    Returns:
        full_count(int): This is the predicted full kernel count calculated by our last trained model
    """
    last_training_dir = f'{len(os.listdir(MODELS_DIR))}'
    model_name        = f'kernel_prediction_model-{ITERATIONS}.meta'

    try:
        saver = tf.train.import_meta_graph(os.path.join(MODELS_DIR, last_training_dir, model_name))
    except Exception as e:
        print(e)
        exit(-1)

    with tf.Session() as session:
        # Load last training module
        saver.restore(session, tf.train.latest_checkpoint(os.path.join(MODELS_DIR, last_training_dir)))

        x  = tf.placeholder(tf.float32,[None,N])
        W  = session.run("W:0")   # Load weights.
        b  = session.run("b:0")   # Load basis.
        y  = tf.matmul(x, W) + b  # Machine learning model.

        # Generate full count.
        feed = {x: [[front_count, ratio]]}
        full_count = session.run(y, feed_dict=feed)
        full_count = int(full_count[0][0])

        return full_count

def generate_training_set():
    """Place's each corn photo's features and final kernel count on a row
        in dataset.csv

    Args:
        None
    Returns:
        None
    """
    try:
        feature_file     = open(csv_features.FILENAME, 'r')
        total_count_file = open('csv/total_kernel_counts.csv', 'r')
        data_file        = open('csv/dataset.csv', 'w+')
    except IOError as e:
        print(e)
        exit(-1)

    feature_reader     = csv.reader(feature_file, delimiter='|', quotechar='/', quoting=csv.QUOTE_MINIMAL)
    total_count_reader = csv.reader(total_count_file , delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)
    data_writer        = csv.writer(data_file,    delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)

    # Call next() to skip past header row.
    next(feature_reader)
    next(total_count_reader)

    # Get first data row of csv file.
    feature_row     = next(feature_reader)
    total_count_row = next(total_count_reader)

    feature_file_end_reached = False;

    while not feature_file_end_reached:
        # Get leading integer of file name. e.i '1-batch1 copy.JPG'
        corn_feature_id = int(feature_row[0].split('-')[0])
        corn_total_id   = int(total_count_row[0])

        # Seek the absolute reader to the current corn feature id
        while corn_total_id < corn_feature_id:
            try:
                total_count_row = next(total_count_reader)
                corn_total_id   = int(total_count_row[0])
            except StopIteration:
                # Consumed all rows in the absolute_features.csv file.
                # This means we're training a corn id we do not have a final count
                # for. The program will be terminated.
                print(f"Final kernel count does not exist for corn ID: {corn_feature_id}.")
                print("Training will now terminate.")
                feature_file.close()
                total_count_file.close()
                data_file.close()
                exit(-1)

        # Extract features and the full kernel count from the csv files.
        front_count = int(feature_row[1])
        w_h_ratio   = float(feature_row[2])
        full_count  = int(total_count_row[3])

        # Write the features and full kernel count to the data file.
        data_writer.writerow([front_count, w_h_ratio, full_count])

        try:
            # Repeat process for the next image in feature_reader.
            feature_row = next(feature_reader)
        except:
            # Reached the end of the feature file, stop iteration.
            feature_file_end_reached = True

    feature_file.close()
    total_count_file.close()
    data_file.close()

def train():
    '''
    Trains our counting model with datapoints from dataset.csv

    Args:
        None
    Returns:
        None
    '''
    try:
        data = np.genfromtxt('csv/dataset.csv', delimiter=',')
    except IOError as e:
        print(e)
        print("dataset.csv will be created after a set of images have been processed.")
        exit(-1)

    x  = tf.placeholder(tf.float32,[None,N])     # Placeholder for N features.
    y_ = tf.placeholder(tf.float32, [None, 1])   # Placeholder for final kernel count.
    W  = tf.Variable(tf.zeros([N,1]), name="W")  # Training for n weights to minimize cost function.
    b  = tf.Variable(tf.zeros([1]),   name="b")  # Training for a b to minimize cost function.
    y  = tf.matmul(x, W) + b                     # Machine learning model.

    cost          = tf.reduce_sum(tf.pow((y_ - y), 2)) # Cost function.
    learning_rate = 0.00001                            # Size of step towards deepest gradient.
    train_step    = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as session:

        session.run(init)
        x_data = None
        y_data = None

        for i in range(1, ITERATIONS + 1):
            for row in data:
                x_data = [row[:N]]    # [ [feature 1, feature 2, feature n] ]
                y_data = [[row[-1]]]  # [ [ final kernel count ] ]
                feed = {x: x_data, y_: y_data}
                session.run(train_step, feed_dict=feed)

        print("\nTraining Finished!")

        # Fetch the trained values
        basis  = (session.run(b))
        weight = (session.run(W))

        # Display to the user the trained values.
        print('=' * 20)
        print(f"Front facing kernel count weight: {weight[0][0]} ")
        print(f"Avg kernel w/h ratio weight:      {weight[1][0]} ")
        print(f"Basis:                            {basis[0]}   ")
        print('=' * 20)
        
        print("Saving trained model...\n")

        # Make folder is MODELS_DIR
        dir_count          = len(os.listdir(MODELS_DIR))
        current_model_dir  = dir_count + 1 #saving dir name as number. Will be using last trained model to get a full count
        model_name         = 'kernel_prediction_model'

        os.makedirs(f'models/{current_model_dir}')

        tf.train.Saver().save(session, os.path.join(MODELS_DIR, f'{current_model_dir}', model_name), global_step=ITERATIONS)

        print("Trained model has been saved.\n")

def main():
    generate_training_set()

if __name__ == "__main__":
    main()