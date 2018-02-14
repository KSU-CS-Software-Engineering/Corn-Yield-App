import numpy as np
import tensorflow as tf
import json

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