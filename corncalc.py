import sys

__version__ = '0.1.0'

def main():
    """
    Entry function of the application
    Currently only displays the version number if the -v flag is used as
    a command line argument to show proof of concept
    """
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
            print(f'Current version is {__version__}')
    else:
        print('No recognizable command or flag was entered')

if __name__ == '__main__':
    main()