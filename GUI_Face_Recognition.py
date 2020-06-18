import tkinter as tk
from tkinter import messagebox
import cv2
import os 
from PIL import Image
import numpy as np
import mysql.connector

window=tk.Tk()
window.title("Face Recognition System by JONATHAN RAI")
#window.config(background="lime")

l1=tk.Label(window,text="Name",font=("Algerian",20))
l1.grid(column=0, row=0)
t1=tk.Entry(window,width=50,bd=5)
t1.grid(column=1,row=0)

l2=tk.Label(window,text="Age",font=("Algerian",20))
l2.grid(column=0, row=1)
t2=tk.Entry(window,width=50,bd=5)
t2.grid(column=1,row=1)

l3=tk.Label(window,text="Address",font=("Algerian",20))
l3.grid(column=0, row=2)
t3=tk.Entry(window,width=50,bd=5)
t3.grid(column=1,row=2)


def train_classifier():
        data_dir="C:/Users/Jonathan/Desktop/Face Recognizer/data" 
        #forward slash ma change gareko
        path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        #stored img ko path lai pass garnae data_dir ko rup ma and and join my data to f
        #f means image, f means listdir(data_dir) 
        # we'll pass data_dir to func os.lis wala and it'l list my all img 
        # and it'll join my list to tyo join(data_dir,f)
        # tyo path ma \1.1 jpg tesari basdai janxa
        # tyo for f bhaneko m img in data_dir directory
        faces = []
        ids = []

        for image in path:
            img = Image.open(image).convert('L');
            #to convert img to gray, this is another img
            imageNp = np.array(img, 'uint8') #u int 8 is a type
            id = int(os.path.split(image)[1].split(".")[1])
            #split gareko c:\Users yesto lai split gareko
            #C:\Users\Jonathan\Desktop\Face Recognizer\data\user.1.1.jpg
            # suru dekhi data samma 0 index tespaxi user dekhi 1 index bhaneko
            #1.1 ma agadee ko 1 index 1 ma and second wala 1 second index ma
            # agadee ko 1.1 ko 1 chahee id ma janxa

            faces.append(imageNp)
            ids.append(id) # mathi ko id = os. wala ra ids[] append gareko
        ids = np.array(ids) #converting ids to array format

        #Train the classifier and save
        clf = cv2.face.LBPHFaceRecognizer_create()
        # using LBPH face recognizer to train the classifier
        clf.train(faces,ids) #passing faces, and ids to classifier
        clf.write("classifier.xml") # to save
        messagebox.showinfo('Result','Training dataset completed!!!')

    
b1=tk.Button(window,text="Training", font=("Algerian",20),bg='orange',fg='red', command=train_classifier)
b1.grid(column=0, row=4)



def detect_face():
    def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text,clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
        # I will not pass value over here, I will pass above parameter

        coords = [] 

        for(x,y,w,h) in features:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            #Draw rectangle to RGB image or live video from webcam so I have to pass
            #real img not gray image 2 is thickness value
            #To predict my image, I have to crop my real image and convert to gray image
            # so draw rectangle to real image
            id,pred = clf.predict(gray_image[y:y+h,x:x+w]) # predict my face from gra scale image as g image in dataset
            #tyo y: y+h crop gareko # classifier lai deko to predict id and pred value
            # id is the prediction value, 1 ho 1.1 ko and we'll use this to
            #calculate confidence percentage for similarity
            confidence = int(100*(1-pred/300)) #Formula
            # this value will show whether authorize img or not
            
            mydb=mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="Authorized_user"
            )
            mycursor=mydb.cursor()
            mycursor.execute("select name from my_table where id="+str(id))
            s = mycursor.fetchone()
            s = ''+''.join(s) # convert name from tuple to string 

            if confidence>75: # if id==1:             #if classifier predict id is 1 then
                 cv2.putText(img,s,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
                #75 % huna paryo
               
               
                #0.8 Font scale value then color value, then thickness,then style of boundary line
                #cursor lai FONT_HERSHEY ma lagera shift + tab thixda parameter format dekhauxa
                # if id==2:      #if classifier predict id is 1 then
                # cv2.putText(img,"Another User Name",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
            else:
                
                cv2.putText(img,"UNKNOWN",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)
                      #0,0,255 bhaneko red color ho BGR anusar

                coords=[x,y,w,h]
        return coords # return coordinates  

    def recognize(img,clf,faceCascade):
        coords = draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face",clf)
        # faceCascade means classifier, means higher cascade from the face
        # 1.1 - Scale Factor, 10 - min neighbor, color is white 255,255,255
        #rename it as face and pass same classifier clf
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # clf - to train the classifier and save 2nd part ma xa
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0) #open camera from laptop

    while True:
        ret, img = video_capture.read() # return image
        img = recognize(img,clf,faceCascade) #get my img from webcam
        cv2.imshow("Face Detection",img)

        if cv2.waitKey(1)==13:
            break
    video_capture.release()
    cv2.destroyAllWindows()

#crop face, convert it to gray image -----> classifier
# To draw rectangle, I have to give my real image that comes from my webcam



b2=tk.Button(window,text="Detect the face", font=("Algerian",20),bg='green',fg='white',command=detect_face)
b2.grid(column=1, row=4)

def generate_dataset():
    
    if(t1.get()=="" or t2.get()=="" or t3.get()==""):
        messagebox.showinfo('Result','Please provide complete details of the user')
    else:
        mydb=mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="Authorized_user"
        )
        mycursor=mydb.cursor()
        #to count no of row in table
        mycursor.execute("SELECT * from my_table")
        myresult=mycursor.fetchall()
        id=1
        for x in myresult:
            id+=1 #yaha id = 0 lekhyo bhanae tala val =(id ma id+1 rakhnu parxa)
        
        sql="insert into my_table(id,Name,Age,Address) values(%s,%s,%s,%s)"
        val=(id,t1.get(),t2.get(),t3.get()) #to count row in table, suppose 3 row then id as 4
        mycursor.execute(sql,val)
        mydb.commit()
        
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #Converts RGB color to gray
            faces = face_classifier.detectMultiScale(gray,1.3,5)
            #1.3 is scaling factor
            #Minimum neighbour = 5 upto what much neighbour you are going to detect face

            if faces is ():
                return None
            for(x,y,w,h) in faces:
                cropped_face=img[y:y+h,x:x+w] #Cropping images for only face part, h means height and w means width
            return cropped_face    

        cap = cv2.VideoCapture(0) #Laptop ko camera so value 0, aru bhaye 1 or -1
        id=1 #id of first authorised person
        img_id=0 # img id means no of image of each authorised person

        while True:
            ret, frame = cap.read() #frame means our img
            if face_cropped(frame) is not None:   # yo frame pass garnae func ma mathi and then returns cropped face
                img_id+=1 # arko user ko lagee id=2 rakhera garnae
                face = cv2.resize(face_cropped(frame),(200,200)) #200 rows and columns
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg" #img ko name user 1.1 tesari xa and convert into jpeg formatt
                cv2.imwrite(file_name_path,face) #store my face in that file_name_path
                cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),2)
                # (50,50) is the origin point where text is to be written
                # font-scale =1 # 0,255,0 is the color
                #thickness = 2

                cv2.imshow("Cropped face",face)
                if cv2.waitKey(1)==13 or int(img_id)==200: # 13 means ASCII value of Enter key and 27 is for Esc key
                    break
        cap.release()    
        cv2.destroyAllWindows()
        messagebox.showinfo('Result','Generating dataset completed!!')


b3=tk.Button(window,text="Generate dataset", font=("Algerian",20),bg='pink',fg='black',command=generate_dataset)
b3.grid(column=2, row=4)

window.geometry("800x200")
window.mainloop()