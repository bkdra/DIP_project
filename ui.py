from window import Ui_MainWindow
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
from fittingRoom import FittingRoom

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.ft = FittingRoom()
        self.pictureModeBtn.clicked.connect(self.pictureModeBtnClicked)
        self.cameraModeBtn.clicked.connect(self.cameraModeBtnClicked)
        self.selectClothbtn.clicked.connect(self.selectClothBtnClicked)
        self.selectPantsbtn.clicked.connect(self.selectPantsBtnClicked)
        self.selectSunglassesbtn.clicked.connect(self.selectSunglassesBtnClicked)
        self.pantSizeSlider.valueChanged.connect(self.pantSizeSliderChanged)
        self.clothSizeSlider.valueChanged.connect(self.clothSizeSliderChanged)
        self.sunglassessize.valueChanged.connect(self.sunglassessizeChanged)
        self.selectBGBtn.clicked.connect(self.selectBGBtnClicked)
        self.BGcheckbox.stateChanged.connect(self.BGcheckboxChanged)
        self.clothCheckBox.stateChanged.connect(self.clothCheckBoxChanged)
        self.pantsCheckBox.stateChanged.connect(self.pantsCheckBoxChanged)
        self.sunglassesCheckBox.stateChanged.connect(self.sunglassesCheckBoxChanged)
        self.pencilStyleBtn.clicked.connect(self.set_pencilStyle)
        self.cartoonStyleBtn.clicked.connect(self.set_cartoonStyle)
        self.set_style()
        
    
    def pictureModeBtnClicked(self):
        self.pictureModeBtn.setChecked(True)
        self.cameraModeBtn.setChecked(False)
        self.clothCheckBox.setChecked(True)
        self.pantsCheckBox.setChecked(True)
        self.sunglassesCheckBox.setChecked(True)
        if hasattr(self, 'camera') and self.camera.isOpened():
            self.camera.release()  # 釋放相機資源
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.ft.picPath = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.update_picture()
        except:
            pass

    def cameraModeBtnClicked(self):
        self.pictureModeBtn.setChecked(False)
        self.cameraModeBtn.setChecked(True)
        self.clothCheckBox.setChecked(True)
        self.pantsCheckBox.setChecked(True)
        self.sunglassesCheckBox.setChecked(True)
        if hasattr(self, 'camera') and self.camera.isOpened():
            self.camera.release()  # 釋放相機資源
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.label.setText("攝影機未開啟")
            return
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)  # 每秒30幀
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            try:
                processed_frame = self.ft.main_capture(frame)
                # 將 OpenCV 影像轉換為 QImage
                cvRGBImg = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(cvRGBImg.data,cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.imageLabel.setPixmap(pixmap)
            except:
                pass

    def selectClothBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.ft.set_cloth(path)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        except:
            pass

    def selectPantsBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.ft.set_pants(path)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        except:
            pass
    
    def selectSunglassesBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.ft.set_sunglass(path)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        except:
            pass

    def pantSizeSliderChanged(self):
        self.ft.set_pantsSize(self.pantSizeSlider.value()/100.0)
        self.pantslabel.setText("調整下著尺寸("+str(self.pantSizeSlider.value())+"%)")
        if self.pictureModeBtn.isChecked():
            self.update_picture()

    def clothSizeSliderChanged(self):
        self.ft.set_clothSize(self.clothSizeSlider.value()/100.0)
        self.clothlabel.setText("調整上著尺寸("+str(self.clothSizeSlider.value())+"%)")
        if self.pictureModeBtn.isChecked():
            self.update_picture()

    def sunglassessizeChanged(self):
        self.ft.set_sunglassesSize(self.sunglassessize.value()/100.0)
        self.sunglasseslabel.setText("調整眼鏡尺寸("+str(self.sunglassessize.value())+"%)")
        if self.pictureModeBtn.isChecked():
            self.update_picture()
    
    def update_picture(self):
        img = self.ft.main_picture()
        cvRGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data,cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.imageLabel.setPixmap(pixmap)
    
    def selectBGBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.ft.set_background(path)
            self.BGcheckbox.setChecked(True)
            self.ft.backgroundisON = True
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        except:
            pass

    def BGcheckboxChanged(self):
        if self.BGcheckbox.isChecked():
            self.ft.backgroundisON = True
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        else:
            self.ft.backgroundisON = False
            if self.pictureModeBtn.isChecked():
                self.update_picture()

    def clothCheckBoxChanged(self):
        if self.clothCheckBox.isChecked():
            self.ft.clothisON = True
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        else:
            self.ft.clothisON = False
            if self.pictureModeBtn.isChecked():
                self.update_picture()
    
    def pantsCheckBoxChanged(self):
        if self.pantsCheckBox.isChecked():
            self.ft.pantsisON = True
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        else:
            self.ft.pantsisON = False
            if self.pictureModeBtn.isChecked():
                self.update_picture()
    
    def sunglassesCheckBoxChanged(self):
        if self.sunglassesCheckBox.isChecked():
            self.ft.sunglassesisON = True
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        else:
            self.ft.sunglassesisON = False
            if self.pictureModeBtn.isChecked():
                self.update_picture()

    def set_pencilStyle(self):
        if self.pencilStyleBtn.isChecked():
            self.ft.style = 1
            self.cartoonStyleBtn.setChecked(False)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        else:
            self.ft.style = 0
            if self.pictureModeBtn.isChecked():
                self.update_picture()

    def set_cartoonStyle(self):
        if self.cartoonStyleBtn.isChecked():
            self.ft.style = 2
            self.pencilStyleBtn.setChecked(False)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        else:
            self.ft.style = 0
            if self.pictureModeBtn.isChecked():
                self.update_picture()



    def set_style(self):
        SBstylesheet = '''
            QScrollBarArea{
                border: 0px;
           }
            QScrollBar:vertical
            {
                background : solid rgb(31,41,55);
            }

            QScrollBar::handle:vertical
            {
                background-color: rgb(50, 130, 184)
            }

            QScrollBar::groove:vertical
            {
                background-color: rgb(31,41,55);
            }
            QScrollBar:horizontal
            {
                background : solid rgb(31,41,55);
            }

            QScrollBar::handle:horizontal
            {
                background-color: rgb(50, 130, 184)
            }

            QScrollBar::groove:horizontal
            {
                background-color: rgb(31,41,55);
            }
            ''' 
        self.imageArea.setStyleSheet(SBstylesheet)
        modeBtnstylesheet = '''
            QPushButton{
                color: rgb(0, 0, 0);
                border-color: rgb(187, 225, 250);
                border: 2px solid rgb(187, 225, 250);
                border-top-left-radius :7px;
                background-color: rgb(187, 225, 250);
                border-top-right-radius : 7px;                  
                border-bottom-left-radius : 7px;
                border-bottom-right-radius : 7px
            }
            QPushButton:hover{
                background-color: rgb(50, 130, 184);
                border: 2px solid rgb(50, 130, 184);
            }
            QPushButton:checked{
                background-color: rgb(15, 76, 117);
                border: 2px solid rgb(15, 76, 117);
            }
        '''
        self.pictureModeBtn.setStyleSheet(modeBtnstylesheet)
        self.cameraModeBtn.setStyleSheet(modeBtnstylesheet)
        self.pencilStyleBtn.setStyleSheet(modeBtnstylesheet)
        self.cartoonStyleBtn.setStyleSheet(modeBtnstylesheet)

        selectBtnstylesheet = '''
            QPushButton{
                color: rgb(0, 0, 0);
                border-color: rgb(187, 225, 250);
                border: 2px solid rgb(187, 225, 250);
                border-top-left-radius :7px;
                background-color: rgb(187, 225, 250);
                border-top-right-radius : 7px;                  
                border-bottom-left-radius : 7px;
                border-bottom-right-radius : 7px
            }
            QPushButton:hover{
                background-color: rgb(50, 130, 184);
                border: 2px solid rgb(50, 130, 184);
            }
        '''
        self.selectClothbtn.setStyleSheet(selectBtnstylesheet)
        self.selectPantsbtn.setStyleSheet(selectBtnstylesheet)
        self.selectSunglassesbtn.setStyleSheet(selectBtnstylesheet)
        self.selectBGBtn.setStyleSheet(selectBtnstylesheet)
    