# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
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
from PySide6.QtWidgets import (QApplication, QGraphicsView, QLabel, QMainWindow,
    QMenuBar, QSizePolicy, QTabWidget, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1169, 667)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 610, 21, 31))
        font = QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.playerA_type = QLabel(self.centralwidget)
        self.playerA_type.setObjectName(u"playerA_type")
        self.playerA_type.setGeometry(QRect(150, 10, 131, 31))
        font1 = QFont()
        font1.setPointSize(14)
        self.playerA_type.setFont(font1)
        self.playerA_type.setTextFormat(Qt.PlainText)
        self.playerA_type.setWordWrap(True)
        self.playerB_type = QLabel(self.centralwidget)
        self.playerB_type.setObjectName(u"playerB_type")
        self.playerB_type.setGeometry(QRect(30, 610, 131, 31))
        self.playerB_type.setFont(font1)
        self.playerB_type.setTextFormat(Qt.AutoText)
        self.playerB_type.setWordWrap(True)
        self.field = QGraphicsView(self.centralwidget)
        self.field.setObjectName(u"field")
        self.field.setGeometry(QRect(10, 50, 311, 551))
        self.field.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.field.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        brush = QBrush(QColor(25, 137, 100, 255))
        brush.setStyle(Qt.SolidPattern)
        self.field.setBackgroundBrush(brush)
        self.score = QLabel(self.centralwidget)
        self.score.setObjectName(u"score")
        self.score.setGeometry(QRect(20, 10, 131, 31))
        self.score.setFont(font)
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(290, 10, 21, 31))
        self.label_2.setFont(font)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(340, 10, 811, 631))
        self.tabWidget.setAutoFillBackground(True)
        self.ball_round = QLabel(self.centralwidget)
        self.ball_round.setObjectName(u"ball_round")
        self.ball_round.setGeometry(QRect(180, 610, 141, 31))
        self.ball_round.setFont(font)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1169, 22))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(-1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"B", None))
        self.playerA_type.setText(QCoreApplication.translate("MainWindow", u"unknown", None))
        self.playerB_type.setText(QCoreApplication.translate("MainWindow", u"unknown", None))
        self.score.setText(QCoreApplication.translate("MainWindow", u"A --:-- B", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"A", None))
        self.ball_round.setText(QCoreApplication.translate("MainWindow", u"Round:1", None))
    # retranslateUi

