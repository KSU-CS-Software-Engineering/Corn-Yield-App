import argparse
import corn_app.contour
import corn_app.mask

def main():
	print('Hello, this is a test.')


parser = argparse.ArgumentParser(description='Applies a mask and contours to pictures of corn en masse.', prog="Corn Kernel Contour Application.")
parser.add_argument('--version', action='version', version='Version 0.1.0')
args = parser.parse_args()

if __name__ == '__main__':
	main()
