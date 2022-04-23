import time,sys
import RPi.GPIO as GPIO


class ControlCar:
	def __init__(self):

		#motor control pins
		self.ma_dir = 25
		self.ma_en = 10
		self.mb_dir = 23
		self.mb_en = 24
		self.go_sleep = 7


		#motor driver interrupt
		self.nFault = 12

		#led control
		self.led = 8

		#pin setup
		self.motor_outputs=[self.ma_dir,self.ma_en,self.mb_dir,self.mb_en,self.go_sleep]

		GPIO.setwarnings(False)
		GPIO.setmode(GPIO.BCM)

		for i in self.motor_outputs:
			GPIO.setup(i, GPIO.OUT)

		GPIO.setup(self.led, GPIO.OUT)
		GPIO.setup(self.nFault, GPIO.IN)
		#	Setting PWM outputs
		self.ma_en_pwm = GPIO.PWM(self.ma_en,100)
		self.mb_en_pwm = GPIO.PWM(self.mb_en,100)
		self.ma_en_pwm.start(0)
		self.mb_en_pwm.start(0)
	def turn(self, ma_pwm, mb_pwm, dir_a, dir_b):

		GPIO.output(self.go_sleep,True)

		self.ma_en_pwm.ChangeDutyCycle(ma_pwm)
		GPIO.output(self.ma_dir,dir_a)

		self.mb_en_pwm.ChangeDutyCycle(mb_pwm)
		GPIO.output(self.mb_dir, dir_b)

	def stop(self):
		for i in self.motor_outputs:
			GPIO.output(i,False)
			
test = ControlCar()

if __name__ == '__main__' and False: #Movement reference
	print("All good")
	test.turn(100,100, False, False) #Rotate left
	time.sleep(2)
	test.turn(100,100, True, True) #Rotate Right
	time.sleep(2)
	test.turn(100,100, True, False) #Forward
	time.sleep(2)
	test.turn(100,100, False, True) #Reverse
	time.sleep(2)
	test.turn(50,100, True, False) #Soft left forward
	time.sleep(2)
	test.turn(50,100, False, True) #Soft left reverse
	time.sleep(2)
	test.turn(100,50, True, False) #Soft right forward
	time.sleep(2)
	test.turn(100,50, False, True) #Soft right reverse
	time.sleep(2)
	test.turn(0, 0, True, False)
if False: #Precision testing for "Assault function" Score 9/10
	test.turn(100,100, True, False) #Forward Alignment: Score 9/10
	time.sleep(1) #0.4-0.45 ~ 0.425 m / second
	test.turn(100,100, False, True) #Reverse Alignment: Score 9/10
	time.sleep(1.075)
	test.turn(100,100, True, False) #Forward breaking
	time.sleep(0.075)
	test.turn(0, 0, True, False)
for i in range(1, 11): #Precision testing for "Rotate Step"
	test.turn(100,100, False, False) #Rotate left
	time.sleep(0.5) # ~80 degrees / second (First time only*)
	test.turn(0, 0, True, False)
	time.sleep(0.25)
	#test.turn(100,100, True, True) #Rotate Right
	#time.sleep(0.45) # 
	#test.turn(0, 0, True, False)
	# * On the loop it does 30 degrees per step except for first step which is 45 degrees
	
	test.stop()
