import cv2
import numpy as np
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from rembg import remove



# 顯示圖片
def ShowIMG(img, winName):
    cv2.imshow(winName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 讀圖片
def ReadIMG(fileName):
    img = cv2.imread(fileName)
    return img


# 將衣服的背景去掉
def RemoveBG_cloth(img):
    img_Out = remove(img)
    img_Out = img_Out[:, :, :3]
    return img_Out

# 換背景
def ChangeBG(img, img_BG):
    segmentor = SelfiSegmentation()
    img_BG = cv2.resize(img_BG, (img.shape[1], img.shape[0]))
    result_img = segmentor.removeBG(img, img_BG, cutThreshold=0.7)
    return result_img

# 把圖片的黑邊去掉
def Remove_BlackEdge(grayIMG, img):
    all_y, all_x = np.where(grayIMG == 255)
    
    buttom = min(all_y)
    top = max(all_y)
    left = min(all_x)
    right = max(all_x)

    img = img[buttom:top, left:right]
    return img


# 把衣服加到圖片中的人身上，利用尋找人體中心點對上衣服中心點來加上衣服
def Add_cloth(img, add_img, position1, position2, position3, position4, adjustment = 1):
    # position1: (12.x, 12.y) position2: (11.x, 11.y) position3: (24.x, 24.y) position4: (23.x, 23.y)

    y_offset = int(((position1[1] + position2[1]) / 2 -(position3[1] + position4[1]) / 2)*0.073)
    # 衣服對人在y方向上的偏移量
    middle_position_img = (int((position1[0] + position2[0])/2), int((position1[1] + position2[1] + position3[1] + position4[1]) / 4) + y_offset)
    # 衣服對準的人體的位置
    ret, img_bi = cv2.threshold(add_img, 1, 255, cv2.THRESH_BINARY)
    addIMG_gray = cv2.cvtColor(img_bi, cv2.COLOR_BGR2GRAY)
    add_img = Remove_BlackEdge(addIMG_gray, add_img)
    img_bi = Remove_BlackEdge(addIMG_gray, img_bi)
    # 將去除背景後的衣服的黑邊去掉

    y_size = int(abs((position1[1] + position2[1])/2 - (position3[1] + position4[1])/2) * 1.2 * adjustment)
    if y_size == 0 or y_size == 1:
        y_size = 2
    x_size = int((y_size/add_img.shape[0])*add_img.shape[1])
    add_img = cv2.resize(add_img, (x_size, y_size))
    # 計算衣服在x與y方向的size，並套用在衣服圖片上
    middle_position_addimg = (int(add_img.shape[1]/2), int(add_img.shape[0]/2))
    # 衣服的中心點
    
    ret, img_bi = cv2.threshold(add_img, 1, 255, cv2.THRESH_BINARY)
    img_bi = ~img_bi
    # 獲取衣服的黑白圖(衣服的部分為黑色，其餘部分為白色)
    img_bi = cv2.medianBlur(img_bi, 5)
    # 將沒去除乾淨的噪點去除
    

    # 衣服的圖片超出視窗範圍時的特殊處理，若超出範圍就把多餘的部分切除
    middle_img_diff_x = middle_position_img[0] - middle_position_addimg[0]
    if middle_img_diff_x <= 0:
        middle_img_diff_x = 0
    middle_img_sum_x = middle_position_img[0] + middle_position_addimg[0]
    middle_img_diff_y = middle_position_img[1] - middle_position_addimg[1]
    if middle_img_diff_y <= 0:
        middle_img_diff_y = 0
    middle_img_sum_y = middle_position_img[1] + middle_position_addimg[1]

    temp_img = img[middle_img_diff_y : middle_img_sum_y, middle_img_diff_x : middle_img_sum_x, 0]
    if (abs(temp_img.shape[1] - img_bi.shape[1]) <= 2 and middle_img_diff_y == 0):      # 僅有y超出視窗上側
        img_bi = cv2.resize(img_bi ,(temp_img.shape[1], img_bi.shape[0]))
        add_img = cv2.resize(add_img ,(temp_img.shape[1], add_img.shape[0]))
        img_bi = img_bi[img_bi.shape[0] - temp_img.shape[0]:img_bi.shape[0], :temp_img.shape[1], :]
        add_img = add_img[add_img.shape[0] - temp_img.shape[0]:add_img.shape[0], :temp_img.shape[1], :]
    elif (abs(temp_img.shape[1] - img_bi.shape[1]) <= 2):                               # 僅有y超出視窗下側
        img_bi = cv2.resize(img_bi ,(temp_img.shape[1], img_bi.shape[0]))
        add_img = cv2.resize(add_img ,(temp_img.shape[1], img_bi.shape[0]))
        img_bi = img_bi[:temp_img.shape[0], :temp_img.shape[1], :]
        add_img = add_img[:temp_img.shape[0], :temp_img.shape[1], :]
    elif (abs(temp_img.shape[0] - img_bi.shape[0]) <= 2 and middle_img_diff_x == 0):    # 僅有x超出視窗左側
        img_bi = cv2.resize(img_bi ,(img_bi.shape[1], temp_img.shape[0]))
        add_img = cv2.resize(add_img ,(img_bi.shape[1], temp_img.shape[0]))
        img_bi = img_bi[:temp_img.shape[0], img_bi.shape[1] - temp_img.shape[1]:img_bi.shape[1], :]
        add_img = add_img[:temp_img.shape[0], add_img.shape[1] - temp_img.shape[1]:add_img.shape[1], :]
    elif (abs(temp_img.shape[0] - img_bi.shape[0]) <= 2):                               # 僅有x超出視窗右側
        img_bi = cv2.resize(img_bi ,(img_bi.shape[1], temp_img.shape[0]))
        add_img = cv2.resize(add_img ,(img_bi.shape[1], temp_img.shape[0]))
        img_bi = img_bi[:temp_img.shape[0], :temp_img.shape[1], :]
        add_img = add_img[:temp_img.shape[0], :temp_img.shape[1], :]
    else:                                                                               # 同時超出x和y的邊界
        if (middle_img_sum_x >= img.shape[1]) and (middle_img_sum_y >= img.shape[0]):   # 超出右下
            img_bi = img_bi[:temp_img.shape[0], :temp_img.shape[1], :]
            add_img = add_img[:temp_img.shape[0], :temp_img.shape[1], :]
        elif (middle_img_diff_x == 0) and (middle_img_sum_y >= img.shape[0]):           # 超出左下
            img_bi = img_bi[:temp_img.shape[0], img_bi.shape[1] - temp_img.shape[1]:img_bi.shape[1], :]
            add_img = add_img[:temp_img.shape[0], add_img.shape[1] - temp_img.shape[1]:add_img.shape[1], :]
        elif (middle_img_sum_x >= img.shape[1]) and middle_img_diff_y == 0:             # 超出右上
            img_bi = img_bi[img_bi.shape[0] - temp_img.shape[0]:img_bi.shape[0], :temp_img.shape[1], :]
            add_img = add_img[img_bi.shape[0] - temp_img.shape[0]:img_bi.shape[0], :temp_img.shape[1], :]
        else:                                                                           # 超出左上
            img_bi = img_bi[img_bi.shape[0] - temp_img.shape[0]:img_bi.shape[0], img_bi.shape[1] - temp_img.shape[1]:img_bi.shape[1], :]
            add_img = add_img[img_bi.shape[0] - temp_img.shape[0]:img_bi.shape[0], add_img.shape[1] - temp_img.shape[1]:add_img.shape[1], :]
    
    # 將計算好的衣服貼到圖片中的人物身上
    img_segment = cv2.bitwise_and(img[middle_img_diff_y : middle_img_sum_y, middle_img_diff_x : middle_img_sum_x, :], img_bi)
    img_segment = cv2.bitwise_or(img_segment, add_img)
    img[middle_img_diff_y : middle_img_sum_y, middle_img_diff_x : middle_img_sum_x, :] = img_segment

    # cv2.circle(img, (middle_position_img[0], middle_position_img[1]), 3, (255, 0, 0), 3)
    return img

# 把褲子加到圖片中的人身上
def Add_pants(img, add_img, position1, position2, position3, position4, adjustment = 1):
    # position1: (12.x, 12.y) position2: (11.x, 11.y) position3: (24.x, 24.y) position4: (23.x, 23.y), adjustment對衣服大小進行微調

    x_pos = int((position3[0]+position4[0]) / 2)
    y_pos = int((position3[1]+position4[1])/2 + ((position1[1]+position2[1])/2 - (position3[1]+position4[1])/2)*0.23)

    ret, img_bi = cv2.threshold(add_img, 1, 255, cv2.THRESH_BINARY)
    addIMG_gray = cv2.cvtColor(img_bi, cv2.COLOR_BGR2GRAY)
    add_img = Remove_BlackEdge(addIMG_gray, add_img)
    img_bi = Remove_BlackEdge(addIMG_gray, img_bi)

    x_size = int(abs(position1[0] - position2[0])*1.08*adjustment)             # 褲子x方向上的大小
    if x_size == 0 or x_size == 1:
        x_size = 2
    y_size = int((x_size/add_img.shape[1]) * add_img.shape[0])                 # 褲子y方向上的大小
    add_img = cv2.resize(add_img, (x_size, y_size))
    img_bi = cv2.resize(img_bi, (x_size, y_size))
    img_bi = ~img_bi
    img_bi = cv2.medianBlur(img_bi, 5)

    # 褲子圖片超出螢幕外的特殊處理
    middle_img_diff_x = x_pos-int(x_size/2)
    if middle_img_diff_x <= 0:
        middle_img_diff_x = 0
    middle_img_sum_x = x_pos+int(x_size/2)
    temp_img = img[y_pos : y_pos + y_size, middle_img_diff_x : middle_img_sum_x, :]
    if (middle_img_sum_x >= img.shape[1]) and (add_img.shape[0] == temp_img.shape[0]):          # 僅右側超出螢幕外
        img_bi = img_bi[:temp_img.shape[0], :temp_img.shape[1], :]
        add_img = add_img[:temp_img.shape[0], :temp_img.shape[1], :]
    elif (middle_img_diff_x == 0) and (add_img.shape[0] == temp_img.shape[0]):                  # 僅左側超出螢幕外
        img_bi = img_bi[:temp_img.shape[0], img_bi.shape[1] - temp_img.shape[1]:img_bi.shape[1], :]
        add_img = add_img[:temp_img.shape[0], add_img.shape[1] - temp_img.shape[1]:add_img.shape[1], :]
    elif abs(temp_img.shape[1] - img_bi.shape[1]) <= 2 and y_pos <= 0:                          # 僅上側超出螢幕外
        add_img = cv2.resize(add_img, (temp_img.shape[1], add_img.shape[0]))
        img_bi = cv2.resize(img_bi, (temp_img.shape[1], img_bi.shape[0]))
        img_bi = img_bi[y_pos + y_size - temp_img.shape[0] : y_pos + y_size, :img_bi.shape[1], :]
        add_img = add_img[y_pos + y_size - temp_img.shape[0] : y_pos + y_size, :img_bi.shape[1], :]
    elif abs(temp_img.shape[1] - img_bi.shape[1]) <= 2 and ((y_pos + y_size) >= img_bi.shape[0]): # 僅下側超出螢幕外
        add_img = cv2.resize(add_img, (temp_img.shape[1], add_img.shape[0]))
        img_bi = cv2.resize(img_bi, (temp_img.shape[1], img_bi.shape[0]))
        img_bi = img_bi[: temp_img.shape[0], :img_bi.shape[1], :]
        add_img = add_img[: temp_img.shape[0], :img_bi.shape[1], :]
    else:                                                                                       # 同時超出x和y的邊界
        if (middle_img_sum_x >= img.shape[1]) and ((y_pos + y_size) >= img_bi.shape[0]):        # 超出右下
            img_bi = img_bi[:temp_img.shape[0], :temp_img.shape[1], :]
            add_img = add_img[:temp_img.shape[0], :temp_img.shape[1], :]
        elif (middle_img_diff_x == 0) and ((y_pos + y_size) >= img_bi.shape[0]):                # 超出左下
            img_bi = img_bi[:temp_img.shape[0], img_bi.shape[1] - temp_img.shape[1]:img_bi.shape[1], :]
            add_img = add_img[:temp_img.shape[0], add_img.shape[1] - temp_img.shape[1]:add_img.shape[1], :]
        elif (middle_img_sum_x >= img.shape[1]) and y_pos <= 0:                                 # 超出右上
            img_bi = img_bi[y_pos + y_size - temp_img.shape[0] : y_pos + y_size, :temp_img.shape[1], :]
            add_img = add_img[y_pos + y_size - temp_img.shape[0] : y_pos + y_size, :temp_img.shape[1], :]
        else:                                                                                   # 超出左上
            img_bi = img_bi[y_pos + y_size - temp_img.shape[0] : img_bi.shape[1] - temp_img.shape[1]:img_bi.shape[1], :]
            add_img = add_img[y_pos + y_size - temp_img.shape[0] : add_img.shape[1] - temp_img.shape[1]:add_img.shape[1], :]

    img_segment = cv2.bitwise_and(img[y_pos : y_pos + y_size, middle_img_diff_x : middle_img_sum_x, :], img_bi)
    img_segment = cv2.bitwise_or(img_segment, add_img)
    img[y_pos : y_pos + y_size, middle_img_diff_x : middle_img_sum_x, :] = img_segment

    return img

def Add_sunglass(img, sunglass, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5)  
    # 尋找臉部位置 

    ret, sunglass_1bit = cv2.threshold(sunglass, 1, 255, cv2.THRESH_BINARY)
    sunglass_gray = cv2.cvtColor(sunglass_1bit, cv2.COLOR_BGR2GRAY)
    sunglass = Remove_BlackEdge(sunglass_gray, sunglass)
    sunglass_1bit = Remove_BlackEdge(sunglass_gray, sunglass_1bit)
    sunglass_1bit = ~sunglass_1bit
    cv2.medianBlur(sunglass_1bit, 15, sunglass_1bit)

    temp = ()
    if(type(faces) != type(temp)):
        for singleFace in faces:
            resize_sunglass = cv2.resize(sunglass, (singleFace[2], int(sunglass_1bit.shape[0]*(singleFace[2] / sunglass_1bit.shape[1]))))
            resize_sunglass_1bit = cv2.resize(sunglass_1bit, (singleFace[2], int(sunglass_1bit.shape[0]*(singleFace[2] / sunglass_1bit.shape[1]))))
            img[singleFace[1]+ int(singleFace[3]*0.27):singleFace[1]+resize_sunglass_1bit.shape[0] + int(singleFace[3]*0.27), singleFace[0]:singleFace[0]+resize_sunglass_1bit.shape[1]] = cv2.bitwise_and(img[singleFace[1]+ int(singleFace[3]*0.27):singleFace[1]+resize_sunglass_1bit.shape[0]+ int(singleFace[3]*0.27), singleFace[0]:singleFace[0]+resize_sunglass_1bit.shape[1]], resize_sunglass_1bit)
            img[singleFace[1]+ int(singleFace[3]*0.27):singleFace[1]+resize_sunglass_1bit.shape[0] + int(singleFace[3]*0.27), singleFace[0]:singleFace[0]+resize_sunglass_1bit.shape[1]] = cv2.bitwise_or(img[singleFace[1]+ int(singleFace[3]*0.27):singleFace[1]+resize_sunglass_1bit.shape[0] + int(singleFace[3]*0.27), singleFace[0]:singleFace[0]+resize_sunglass_1bit.shape[1]], resize_sunglass)


# 讀取相機，由他抓取每一偵的圖片，作為底圖，來添加衣服
def main_capture():
    long_sleeves = ReadIMG("long_sleeves.jpg")
    pants = ReadIMG("pants2.jpg")
    sunglass = ReadIMG("sunglass1.jpg")
    background = ReadIMG("fittingRoom.jpg")
    frame = ReadIMG("full.jpg")
    # 讀取衣服、褲子、太陽眼鏡、背景圖片、包含人物的圖片
    long_sleeves = RemoveBG_cloth(long_sleeves)
    pants = RemoveBG_cloth(pants)
    sunglass = RemoveBG_cloth(sunglass)
    # 去背

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # 讀入臉部辨識模型
    solutionPose = mp.solutions.pose
    pose = solutionPose.Pose()
    # 建立身體辨識的物件
    # mpDraw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    # 開啟相機

    if not cap.isOpened():
        print("camara is not opened")
        exit()
    while True:
        ret, frame = cap.read()
        # 讀入相機資訊，ret為是否讀取成功，frame為從相機讀取到的圖片
        if not ret:
            print("picture is not read")
            break

        # frame = ChangeBG(frame, background)
        # 更改背景

        width = 640
        height = 480
        frame = cv2.resize(frame, ((width, height)))
        # 改變視窗大小
        poseResult = pose.process(frame)
        # 尋找身體各部位位置

        Add_sunglass(frame, sunglass, face_cascade)
        # 添加太陽眼鏡

        # 當有讀取到身體各部位資料時
        if poseResult.pose_landmarks:
            position1 = (poseResult.pose_landmarks.landmark[12].x * width, poseResult.pose_landmarks.landmark[12].y * height)
            position2 = (poseResult.pose_landmarks.landmark[11].x * width, poseResult.pose_landmarks.landmark[11].y * height)
            position3 = (poseResult.pose_landmarks.landmark[24].x * width, poseResult.pose_landmarks.landmark[24].y * height)
            position4 = (poseResult.pose_landmarks.landmark[23].x * width, poseResult.pose_landmarks.landmark[23].y * height)
            # 取得身體四個部位的座標(分別是左肩、右肩、左臀部、右臀部)

            condi_m_x1 = position1[0] >= frame.shape[1]
            condi_l_x2 = position2[0] <= 0
            condi_m_x3 = position3[0] >= frame.shape[1]
            condi_l_x3 = position3[0] <= 0
            condi_m_x4 = position4[0] >= frame.shape[1]
            condi_l_x4 = position4[0] <= 0
            condi_m_y1 = position1[1] >= frame.shape[0]
            condi_m_y2 = position2[1] >= frame.shape[0]
            condi_m_y3 = position3[1] >= frame.shape[0]
            condi_l_y3 = position3[1] <= 0
            condi_m_y4 = position4[1] >= frame.shape[0]
            condi_l_y4 = position4[1] <= 0
            # 各個檢測四個部位的座標是否超出圖片邊界的條件
            
            if not((condi_m_y3 and condi_m_y4) or (condi_l_y3 and condi_l_y4) or (condi_l_x3 and condi_l_x4) or (condi_m_x3 and condi_m_x4)):
                frame = Add_pants(frame, pants, position1, position2, position3, position4)
            if not((condi_m_x1) or (condi_l_x2) or (condi_m_y1 and condi_m_y2) or (condi_l_y3 and condi_l_y4)):
                frame = Add_cloth(frame, long_sleeves, position1, position2, position3, position4)
            # 若未超出邊界，則為人的身體加上衣服與褲子
            # mpDraw.draw_landmarks(frame, poseResult.pose_landmarks, solutionPose.POSE_CONNECTIONS)
        
        cv2.imshow("video", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 讀取圖片做為底圖，來添加衣服
def main_picture():
    long_sleeves = ReadIMG("long_sleeves.jpg")
    pants = ReadIMG("pants2.jpg")
    sunglass = ReadIMG("sunglass1.jpg")
    background = ReadIMG("fittingRoom.jpg")
    frame = ReadIMG("full.jpg")
    # 讀取衣服、褲子、太陽眼鏡、背景圖片、包含人物的圖片
    long_sleeves = RemoveBG_cloth(long_sleeves)
    pants = RemoveBG_cloth(pants)
    sunglass = RemoveBG_cloth(sunglass)
    # 去背

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # 讀入臉部辨識模型
    solutionPose = mp.solutions.pose
    pose = solutionPose.Pose()
    # 建立身體辨識的物件

    # frame = ChangeBG(frame, background)
    # 更改背景

    width = frame.shape[1]
    height = frame.shape[0]
    frame = cv2.resize(frame, ((width, height)))
    # 改變視窗大小
    poseResult = pose.process(frame)
    # 尋找身體各部位位置

    Add_sunglass(frame, sunglass, face_cascade)
    # 將墨鏡加到臉上

    # 當有讀取到身體各部位資料時
    if poseResult.pose_landmarks:
        print("run")
        position1 = (poseResult.pose_landmarks.landmark[12].x * width, poseResult.pose_landmarks.landmark[12].y * height)
        position2 = (poseResult.pose_landmarks.landmark[11].x * width, poseResult.pose_landmarks.landmark[11].y * height)
        position3 = (poseResult.pose_landmarks.landmark[24].x * width, poseResult.pose_landmarks.landmark[24].y * height)
        position4 = (poseResult.pose_landmarks.landmark[23].x * width, poseResult.pose_landmarks.landmark[23].y * height)
        # 取得身體四個部位的座標(分別是左肩、右肩、左臀部、右臀部)

        condi_m_x1 = position1[0] >= frame.shape[1]
        condi_l_x2 = position2[0] <= 0
        condi_m_x3 = position3[0] >= frame.shape[1]
        condi_l_x3 = position3[0] <= 0
        condi_m_x4 = position4[0] >= frame.shape[1]
        condi_l_x4 = position4[0] <= 0
        condi_m_y1 = position1[1] >= frame.shape[0]
        condi_m_y2 = position2[1] >= frame.shape[0]
        condi_m_y3 = position3[1] >= frame.shape[0]
        condi_l_y3 = position3[1] <= 0
        condi_m_y4 = position4[1] >= frame.shape[0]
        condi_l_y4 = position4[1] <= 0
        # 各個檢測四個部位的座標是否超出圖片邊界的條件
            
        if not((condi_m_y3 and condi_m_y4) or (condi_l_y3 and condi_l_y4) or (condi_l_x3 and condi_l_x4) or (condi_m_x3 and condi_m_x4)):
            frame = Add_pants(frame, pants, position1, position2, position3, position4)
        if not((condi_m_x1) or (condi_l_x2) or (condi_m_y1 and condi_m_y2) or (condi_l_y3 and condi_l_y4)):
            frame = Add_cloth(frame, long_sleeves, position1, position2, position3, position4)
        # 若未超出邊界，則為人的身體加上衣服與褲子
        
    ShowIMG(frame, "picture")


main_capture()
# main_picture()