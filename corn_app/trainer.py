import numpy as np
import tensorflow as tf
import json
import os
import csv

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

config = json.load(open('../config.json'))
SAVER = tf.train.Saver()
MODEL_NAME = config['trainingCornDir'] + '/kernel_prediction_model'

DATA_FILE = config['features_file']
DELIMITER = ','

#expects float as param
def get_count(front_count):

    saver = tf.train.import_meta_graph(MODEL_NAME + '-' + str(ITERATIONS) + '.meta')

    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint(config['trainingCornDir']))
        print('model restored')
        bias = session.run(BIAS)
        weight = session.run(WEIGHT)
        full_count = weight * np.array(front_count) + bias

        #Todo: train again with new values
        return full_count

def create_data():

    full_count_file = open(config['fullCount_file'], 'r')
    front_count_file = open(config['frontCount_file'], 'r')
    data_file = open(config['features_file'], 'w')
    image_names = sorted(os.listdir(config['cornPhotoDir']))

    data_writer = csv.writer(data_file, delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)
    front_reader = csv.reader(front_count_file, delimiter='|', quotechar='/', quoting=csv.QUOTE_MINIMAL)
    full_reader = csv.reader(full_count_file, delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)

    print(front_reader)

    next(front_reader)

    front_row = next(front_reader)

    next(full_reader)
    full_row = next(full_reader)

    for file in image_names:
        corn_number = int(file.split('_')[0])
        # front_row = next(front_reader)
        # full_row = next(full_reader)
        # print(front_row)
        # print(full_row)

        corn_number_check = int(front_row[0].split('_')[0])

        while(corn_number != corn_number_check):
            print(corn_number, front_row)

            front_row = next(front_reader)
            corn_number_check = int(front_row[0].split('_')[0])

        front_count = int(front_row[1])

        corn_number_check = int(full_row[0])
        while(corn_number != corn_number_check):
            print(corn_number, full_row)
            full_row = next(full_reader)
            corn_number_check = int(full_row[0].split('_')[0])

        full_count = int(full_row[3])

        data_writer.writerow([front_count, full_count])


def train():

    # get data from csv file
    data_points = np.genfromtxt(DATA_FILE, delimiter=DELIMITER)
    sample_size = len(data_points)

    init = tf.global_variables_initializer()

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
                print("ITERATION: {0}, WEIGHT: {1}, BIAS: {2}, ERROR: {3}".format(number_of_runs, current_weight, current_bias, current_error))

        print("\nTraining Finished!")

        final_error = session.run(ERROR, feed_dict={FRONT_KERNEL_COUNT: coordinates[X], FULL_KERNEL_COUNT: coordinates[Y]})
        final_weight = session.run(WEIGHT)
        final_bias = session.run(BIAS)

        print("After {0} iterations,\n  WEIGHT: {1}, BIAS: {2}, ERROR: {3}".format(ITERATIONS, final_weight, final_bias, final_error))

        SAVER.save(session, MODEL_NAME, global_step=ITERATIONS)

if __name__ == "__main__":
    main()