import utilities_project as utilities
from os import listdir
from os.path import isfile, join



########## Globales
PATH_HTML_FILES = "data/htmlfiles"
info_files = []

########## Functionality

def main():
    dataframe = utilities.pandas_read_csv()
    print("")

if __name__ == "__main__":
    main()
