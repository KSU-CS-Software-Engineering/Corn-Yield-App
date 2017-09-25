import sys

__version__ = '0.1.0'

def main():
    """The entry function of the application.

    Current functionality is a place holder.

    Example: $python3 corn_calc.py -v.
    """
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
            print(f'Current version is {__version__}')
    else:
        print('No recognizable command or flag was entered')

if __name__ == '__main__':
    main()