import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

n_fault=12

mb_di=23
mb_en=24
ma_di=25
ma_en=10
go_sleep=7

control_list=[mb_di,mb_en,ma_di,ma_en,go_sleep]
for i in control_list:
    GPIO.setup(i,GPIO.OUT)

def drive():
    GPIO.output(go_sleep,True)
    
    GPIO.output(ma_en,True)
    GPIO.output(ma_di,True)
    
    GPIO.output(mb_en,True)
    GPIO.output(mb_di, False)
    
    
    

def reverse():
    GPIO.output(go_sleep,True)
    
    GPIO.output(ma_en,True)
    GPIO.output(ma_di,False)
    
    GPIO.output(mb_en,True)
    GPIO.output(mb_di, True)
    
def left():
    GPIO.output(go_sleep,True)
    
    GPIO.output(ma_en,True)
    GPIO.output(ma_di,False)
    
    GPIO.output(mb_en,True)
    GPIO.output(mb_di, False)
    
    
def right():
    GPIO.output(go_sleep,True)
    
    GPIO.output(ma_en,True)
    GPIO.output(ma_di,True)
    
    GPIO.output(mb_en,True)
    GPIO.output(mb_di, True)

    
def stop():
    for i in control_list:
        GPIO.output(i,False)
    

