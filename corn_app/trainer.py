import numpy as np
import tensorflow as tf
import json
import os
import csv
import csv_features

LEARNING_RATE = 0.00001
ITERATIONS = 1000
DISPLAY_INTERVAL = 50

X = 0
Y = 1

FRONT_KERNEL_COUNT = tf.placeholder(tf.float32)
FULL_KERNEL_COUNT = tf.placeholder(tf.float32)
WEIGHT = tf.Variable(tf.random_normal([1]))
BIAS = tf.Variable(tf.random_normal([1]))
TRAINING_MODEL = tf.add(tf.multiply(WEIGHT, FRONT_KERNEL_COUNT), BIAS)
ERROR = tf.reduce_mean(tf.square(TRAINING_MODEL - FULL_KERNEL_COUNT))
TRAINER = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(ERROR)

SAVER = tf.train.Saver()



#expects float as param
def get_count(front_count):
    """
    Uses last trained model to predict the full kernel count

    Args:
        front_count(float): This is the count determined by the watershed/otsu method in the contours module

    Returns:
        full_count(float): This is the predicted full kernel count calculated by our last trained model
    """

    models_dir = '../models'
    last_training_dir = f'{len(os.listdir(models_dir))}'
    model_name = f'kernel_prediction_model-{ITERATIONS}.meta'

    saver = tf.train.import_meta_graph(os.path.join(models_dir, last_training_dir, model_name))

    with tf.Session() as session:
        print('Restoring last training model...\n')
        saver.restore(session, tf.train.latest_checkpoint(os.path.join(models_dir, last_training_dir)))
        print('Last training model restored...\n')

        print('Calculating full kernel count...\n')
        bias = session.run(BIAS)
        weight = session.run(WEIGHT)
        full_count = weight * front_count + bias

        print(f'The predicted full kernel count is : {full_count}\n')

        print('Training model with new values...\n')

        session.run(TRAINER, feed_dict={FRONT_KERNEL_COUNT: front_count, FULL_KERNEL_COUNT: full_count})
        final_error = session.run(ERROR, feed_dict={FRONT_KERNEL_COUNT: front_count, FULL_KERNEL_COUNT: full_count})
        final_weight = session.run(WEIGHT)
        final_bias = session.run(BIAS)

        print("Training Finished!\n")
        print(f"WEIGHT: {final_weight}, BIAS: {final_bias}, ERROR: {final_error}")

        #Todo: save session
        return full_count

def generate_training_set():
    """Place's each corn photo's features and final kernel count on a row
        in dataset.csv

    Args:
        None
    Returns:
        None
    """

    feature_file     = open(csv_features.FILENAME, 'r')
    total_count_file = open('csv/total_kernel_counts.csv', 'r')
    data_file        = open('csv/dataset.csv', 'w+')

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
                exit(0)

        # Extract features and the full kernel count from the csv files.
        front_count = int(feature_row[1])
        full_count  = int(total_count_row[3])

        # Write the features and full kernel count to the data file.
        data_writer.writerow([front_count, full_count])

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

    # get data pairs from csv file
    data_points = np.genfromtxt('../csv/dataset.csv', delimiter=',')

    #will be run later to initialize tensorflow variables
    init = tf.global_variables_initializer()

    #every tensor flow object/function has to be run in the session for persistence
    with tf.Session() as session:

        #initialize all tf variables
        session.run(init)

        for current_run in range(ITERATIONS):

            number_of_runs = current_run + 1

            #train for each front_count => full_count pair
            for coordinates in data_points:

                session.run(TRAINER, feed_dict={FRONT_KERNEL_COUNT: coordinates[X], FULL_KERNEL_COUNT: coordinates[Y]})

            if number_of_runs % DISPLAY_INTERVAL == 0:
                current_error = session.run(ERROR, feed_dict={FRONT_KERNEL_COUNT: coordinates[X], FULL_KERNEL_COUNT: coordinates[Y]})
                current_weight = session.run(WEIGHT)
                current_bias = session.run(BIAS)
                print(f"ITERATION: {number_of_runs}, WEIGHT: {current_weight}, BIAS: {current_bias}, ERROR: {current_error}")

        print("\nTraining Finished!")

        final_error = session.run(ERROR, feed_dict={FRONT_KERNEL_COUNT: coordinates[X], FULL_KERNEL_COUNT: coordinates[Y]})
        final_weight = session.run(WEIGHT)
        final_bias = session.run(BIAS)

        print(f"After {ITERATIONS} iterations,\n  WEIGHT: {final_weight}, BIAS: {final_bias}, ERROR: {final_error}")

        print("Saving trained model...")

        #create folder in the models directory
        models_dir = '../models'
        os.chdir(models_dir)

        dir_count          = len(os.listdir(models_dir))
        current_model_dir  = dir_count + 1 #saving dir name as number. Will be using last trained model to get a full count
        model_name         = 'kernel_prediction_model'

        os.makedirs(f'{current_model_dir}')

        SAVER.save(session, os.path.join(models_dir, f'{current_model_dir}', model_name), global_step=ITERATIONS)

        print("Trained model has been saved.")

def main():
    generate_training_set()

if __name__ == "__main__":
    main()