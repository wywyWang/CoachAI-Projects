# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'training.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(476, 447)
        self.training_code = QLineEdit(Form)
        self.training_code.setObjectName(u"training_code")
        self.training_code.setGeometry(QRect(100, 40, 301, 20))
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 40, 81, 16))
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 70, 61, 20))
        self.training_data = QLineEdit(Form)
        self.training_data.setObjectName(u"training_data")
        self.training_data.setGeometry(QRect(100, 70, 301, 20))
        self.output_path = QLineEdit(Form)
        self.output_path.setObjectName(u"output_path")
        self.output_path.setGeometry(QRect(100, 100, 301, 20))
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 100, 71, 20))
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(30, 140, 81, 16))
        self.max_iteration = QLineEdit(Form)
        self.max_iteration.setObjectName(u"max_iteration")
        self.max_iteration.setGeometry(QRect(120, 140, 71, 20))
        self.learning_rate = QLineEdit(Form)
        self.learning_rate.setObjectName(u"learning_rate")
        self.learning_rate.setGeometry(QRect(120, 180, 71, 20))
        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(30, 180, 81, 16))
        self.confirm = QPushButton(Form)
        self.confirm.setObjectName(u"confirm")
        self.confirm.setGeometry(QRect(310, 220, 75, 24))
        self.load_training_code = QPushButton(Form)
        self.load_training_code.setObjectName(u"load_training_code")
        self.load_training_code.setGeometry(QRect(381, 41, 19, 18))
        self.load_training_data = QPushButton(Form)
        self.load_training_data.setObjectName(u"load_training_data")
        self.load_training_data.setGeometry(QRect(381, 71, 19, 18))
        self.error_message = QLabel(Form)
        self.error_message.setObjectName(u"error_message")
        self.error_message.setGeometry(QRect(112, 220, 191, 20))
        self.error_message.setLayoutDirection(Qt.LeftToRight)
        self.error_message.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label.setText(QCoreApplication.translate("Form", u"training code", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"traing data", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"output path", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"max iteration", None))
        self.max_iteration.setText(QCoreApplication.translate("Form", u"100", None))
        self.learning_rate.setText(QCoreApplication.translate("Form", u"0.001", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"learning rate", None))
        self.confirm.setText(QCoreApplication.translate("Form", u"train", None))
        self.load_training_code.setText(QCoreApplication.translate("Form", u"...", None))
        self.load_training_data.setText(QCoreApplication.translate("Form", u"...", None))
        self.error_message.setText("")
    # retranslateUi

