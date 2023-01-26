import datetime as dt
import logging
import os
from os.path import join
import yaml

##==========================================================================================================
"""
Function:       isfloat()
Description:    Get bool that states whether the sent parameter is float or not
Return:         Boolean - True if isfloat(num) else False
"""
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

##==========================================================================================================
"""
Function:       get_datetime_format()
Description:    Get the date and time with the format for storing files
Return:         String - Date and time
"""
def get_datetime_format():
    return dt.datetime.now().strftime("%Y%m%d%H%M%S")

##==========================================================================================================
"""
Function:       configure_logger()
Description:    Configure logger of the project
Return:         None
"""
def configure_logger(levelStdout=logging.DEBUG, levelFile=logging.DEBUG, path_project=".", path_dir_logs="logs/", _datetime="YYYYMMDD", pattern="binaryClassif"):
    global LOGGER
    
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    """
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(levelStdout)
    stdout_handler.setFormatter(formatter)
    """
    
    file_handler = logging.FileHandler(join(path_project, path_dir_logs, _datetime + '_' + pattern + '.log'))
    
    file_handler.setLevel(levelFile)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    #LOGGER.addHandler(stdout_handler)

    return LOGGER

def read_config_file(config_file_path):
    if config_file_path == None:
        config_file_path = "config.yml"
    
    with open(join(config_file_path), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        return cfg

    return None