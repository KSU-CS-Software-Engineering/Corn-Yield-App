"""Module of constants pertaining to the csv file containing the ear features
Attributes:
    FILENAME  (str): Filename of the ear feature csv file
    HEADER    (str): Header of the csv file
    DELIM     (str): Delimiter character used for the csv file
    QUOTECHAR (str): Character used for quotation marks
"""

FILENAME  = 'calculated_features.csv'
HEADER    = ['image filename', 'front facing kernel count', "avg width/height ratio"]
DELIM     = '|'
QUOTECHAR = '/'