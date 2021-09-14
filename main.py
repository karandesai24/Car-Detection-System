#!/usr/bin/env python
# coding: utf-8

# In[98]:


pip install opencv-python


# In[99]:


import cv2


# In[100]:


car_Classifier=cv2.CascadeClassifier("D:\Car detection\cars.xml")


# In[101]:


vid=cv2.VideoCapture("D:\Car detection\car.mp4")


# In[102]:


while True:
    ret, frame=vid.read()
    if ret == False:
        break
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    car=car_Classifier.detectMultiScale(gray, 1.1 , 1)
    
    for(x,y,w,h) in car:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.imshow('Car Detection', frame)
        
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
    

