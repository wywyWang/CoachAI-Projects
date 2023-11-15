# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3

# To make py runnable inside and outside folder
# For relative imports to work in Python 3.6
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
