# This Python file uses the following encoding: utf-8
import os
import re

from PySide6.QtWidgets import QLineEdit, QFileDialog
from PySide6.QtCore import Qt, Signal, Slot, QThread
import pandas as pd
import ast
import numpy as np

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from GUI.TrainingWidget import TrainingWidget
from PySide6.QtGui import QColor, QDoubleValidator, QPen, QIntValidator

class TrainingWidgetCtrl:
    def __init__(self, widget: TrainingWidget):
        self.parent = widget
        self.ui = widget.ui

        # file input
        self.ui.load_training_code.clicked.connect(lambda: self.setFile(self.ui.training_code,"py檔案(*.py)"))
        self.ui.load_training_data.clicked.connect(lambda: self.setFile(self.ui.training_data,"csv檔案(*.csv)"))


        only_float = QDoubleValidator()
        only_float.setRange(0.0, 5.0)
        self.ui.learning_rate.setValidator(only_float)

        only_int = QIntValidator()
        only_int.setRange(0, 1000000000)
        self.ui.max_iteration.setValidator(only_int)

        self.ui.confirm.clicked.connect(self.preprocessData)


    # event function for selecting file
    @Slot(QLineEdit)
    def setFile(self, target: QLineEdit, filter: str):
        filename, _ = QFileDialog.getOpenFileName(self.parent, 
                                                  caption="Open file", 
                                                  dir= "./", 
                                                  filter=filter)
        if len(filename) == 0:
            return
        
        target.setText(filename)

    @Slot()
    def preprocessData(self):
        if not os.path.exists(self.ui.training_code.text()):
            self.ui.error_message.setText('training code not exist')
            return
        if not os.path.exists(self.ui.training_data.text()):
            self.ui.error_message.setText('training data not exist')
            return
        
        if self.ui.output_path.text() == "":
            self.ui.error_message.setText("output file name can't be empty.")
            return
        
        self.ui.confirm.setEnabled(False)
        self.preprocess_thread = TrainingThread(self.ui.training_code.text(), 
                                                  self.ui.training_data.text(), 
                                                  self.ui.max_iteration.text(),
                                                  self.ui.learning_rate.text(),
                                                  self.ui.output_path.text(),)

        self.preprocess_thread.finished.connect(self.threadFinished)
        self.preprocess_thread.start()

    # event function for preprocess thread to allow next input since current is finish
    @Slot()
    def threadFinished(self):
        self.ui.confirm.setEnabled(True)
        
class TrainingThread(QThread):
    finished = Signal()

    def __init__(self, code:str, data:str, max_iter:str, learning_rate:str, save_path: str, parent=None):
        super().__init__(parent)
        self.code = code
        self.data = data
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.save_path = save_path


    def run(self):
        print(f'{self.code} --data {self.data} --learning_rate {self.learning_rate} --max_iter {self.max_iter} --output {self.save_path}')
        os.system(f'{self.code} --data {self.data} --learning_rate {self.learning_rate} --max_iter {self.max_iter} --output {self.save_path}')
        self.finished.emit()

    