import tkinter as tk
from keras.models import load_model
from tkinter import filedialog
import cv2
import numpy as np
from keras.utils import img_to_array
from PIL import Image, ImageTk, ImageFilter

selected_image_path = None
width = 0
height = 0
face_classifier=cv2.CascadeClassifier('E:/Nam_3/AI/CuoiKy/haarcascade/haarcascade_frontalface_default.xml')
age_model = load_model('E:/Nam_3/AI/CuoiKy/Model/age_model_50epochs_128.h5')
gender_model = load_model('E:/Nam_3/AI/CuoiKy/Model/gender_model_50epochs_128.h5')
logo_ute_path = ("E:/logo - cover MEC/1200px-Hcmute.svg.png")
logo_fme_path = ("E:/logo - cover MEC/fme-1.png")
gender_labels = ['Male', 'Female']


def browse_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename()

def predict_output():
    image = cv2.imread(selected_image_path, cv2.IMREAD_COLOR)
    height, width, _ = image.shape
    if (width > height):
         image = cv2.resize(image, (936, 720))
    else:
        image = cv2.resize(image, (720, 936)) 

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)
        roi_color=image[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(128,128),interpolation=cv2.INTER_AREA)

        #Gender
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,128,128,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 
        gender_label_position=(x,y+h+50) 
        cv2.putText(image,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Age
        age_predict = age_model.predict(np.array(roi_color).reshape(-1,128,128,3))
        age = round((age_predict[0,0]))
        age_label_position=(x,y)
        cv2.putText(image,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow('Age & Gender Detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def real_time_detection():
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            cv2.imshow("Gray",roi_gray)
            roi_gray=cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)
            #Get image ready for prediction
            
            roi_color=frame[y:y+h,x:x+w]
            cv2.imshow("RGB",roi_color)
            roi_color=cv2.resize(roi_color,(128,128),interpolation=cv2.INTER_AREA)

            #Gender
            gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,128,128,3))
            gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
            gender_label=gender_labels[gender_predict[0]] 
            gender_label_position=(x,y+h+50) #50 pixels below to move the label outside the face
            cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            #Age
            age_predict = age_model.predict(np.array(roi_color).reshape(-1,128,128,3))
            age = round((age_predict[0,0]))
            age_label_position=(x,y)
            cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Age & Gender Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def exit_app():
    root.destroy()

# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("AGE & GENDER DETECTION")
#-------------------- hien thi logo-----------------------------
image_ute_frame = tk.Frame(root)
image_ute_frame.place(x=10, y=10)

image_ute = Image.open(logo_ute_path)
image_ute = image_ute.resize((120, 153))
image_ute_tk = ImageTk.PhotoImage(image_ute)

label_ute = tk.Label(image_ute_frame, image=image_ute_tk)
label_ute.pack()

#------------------------Logo FME ------------------------------

image_fme_frame = tk.Frame(root)
image_fme_frame.place(x=580, y=10)

image_fme = Image.open(logo_fme_path)
image_fme = image_fme.resize((125, 125))
image_fme_tk = ImageTk.PhotoImage(image_fme)

label_fme = tk.Label(image_fme_frame, image=image_fme_tk)
label_fme.pack()
#---------------------------------------------------------------
# Hiển thị tiêu đề
label = tk.Label(root, text="TRƯỜNG ĐH SƯ PHẠM KỸ THUẬT TPHCM", fg="red", font=("Arial", 17), anchor="nw")
label.place(x=125, y=5)
label1 = tk.Label(root, text="KHOA CƠ KHÍ CHẾ TẠO MÁY", fg="blue", font=("Arial", 16), anchor="nw")
label1.place(x=220, y=40)
label2 = tk.Label(root, text="BỘ MÔN CƠ ĐIỆN TỬ", fg="black", font=("Arial", 15), anchor="nw")
label2.place(x=260, y=70)
label3 = tk.Label(root, text="PROJECT CUỐI KỲ", fg="black", font=("Arial", 14), anchor="nw")
label3.place(x=275, y=150)
label4 = tk.Label(root, text="MÔN HỌC: TRÍ TUỆ NHÂN TẠO", fg="black", font=("Arial", 14), anchor="nw")
label4.place(x=230, y=190)
label5 = tk.Label(root, text="Age and Gender Detection with Python", fg="red", font=("Arial", 22), anchor="nw")
label5.place(x=125, y=220)

label6 = tk.Label(root, text="SVTH: ", fg="black", font=("Arial", 12), anchor="nw")
label6.place(x=5, y=380)
label7 = tk.Label(root, text="Hồ Vĩnh Trọng  20146107", fg="black", font=("Arial", 12), anchor="nw")
label7.place(x=5, y=410)
label8 = tk.Label(root, text="GVHD: ", fg="black", font=("Arial", 12), anchor="nw")
label8.place(x=5, y=440)
label9 = tk.Label(root, text="PGS.TS Nguyễn Trường Thịnh", fg="black", font=("Arial", 12), anchor="nw")
label9.place(x=5, y=470)


window_width = 720
window_height = 512
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")


# Kích thước nút
button_width = 10
button_height = 2


# Tạo các nút nhấn
btn_browse = tk.Button(root, text="Browse", command=browse_image, font=("Arial", 14), justify=tk.LEFT,
                       width=10, height=1)
btn_browse.place(x=430, y=330, anchor=tk.NW)

btn_predict = tk.Button(root, text="Predict", command=predict_output, font=("Arial", 14), justify=tk.LEFT,
                        width=10, height=1)
btn_predict.place(x=550, y=330, anchor=tk.NW)

btn_real_time = tk.Button(root, text="Real-Time Detection", command=real_time_detection, font=("Arial", 14), justify=tk.LEFT,
                          width=18, height=button_height)
btn_real_time.place(x=450, y=400, anchor=tk.NW)

btn_exit = tk.Button(root, text="Exit", command=exit_app, font=("Arial", 14),bg='red', justify=tk.LEFT,width=button_width, height=button_height)
btn_exit.place(x=280, y=370, anchor=tk.NW)

root.mainloop()
