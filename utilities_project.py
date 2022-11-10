from google.colab import drive
import pandas as pd

#'/content/drive/"Colab Notebooks"/IsaacOlguin/AutomatedTraumaDetectionInGCT'

ROOT_DRIVE_PATH = '/content/drive'

def drive_connect():
    drive.mount(ROOT_DRIVE_PATH)

def drive_connect_to_path(path):
    if path == "":
        drive_connect()
    else:
        drive.mount(path)

def pandas_read_csv(path_file, _delimiter):
    information = pd.Series()
    try:
        information = pd.read_csv(path_file, delimiter=_delimiter)
    except Exception as exc:
        print(f"\nERROR An error occurs while reading CSV file with pandas")

    return information