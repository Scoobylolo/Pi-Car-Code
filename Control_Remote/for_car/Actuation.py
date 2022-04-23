import time,sys
import RPi.GPIO as GPIO
sys.path.append("../")
import mutual
from functools import partial

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

		self.forward_dir=mutual.forward
		self.back_dir=mutual.back
		self.left_dir=mutual.left
		self.right_dir=mutual.right

		self.acceptable_dirs=[self.forward_dir,self.back_dir,self.left_dir,self.right_dir]
		self.functions=[partial(self.turn,100,100,True,False),partial(self.turn,100,100,False,True),partial(self.turn,100,100,False,False),partial(self.turn,100,100,True,True)]
		self.dirs_and_functions=dict(zip(self.acceptable_dirs,self.functions))
		
		self.sleep_time=0.05
		self.consecutive_presses=[]
		self.history_of_keys=[]


	def car_cpu(self,direction):
		direction,state=direction.split(" ")
		if state=="released":
			self.stop()
			
			most_frequent=max(set(self.consecutive_presses), key=self.consecutive_presses.count)
			if most_frequent!="" and len(self.consecutive_presses)>2:
				if most_frequent.split(" ")[0]==self.forward_dir:
					self.reverse()
					time.sleep(self.sleep_time)
					self.stop()
				elif most_frequent.split(" ")[0]==self.back_dir:
					self.drive()
					time.sleep(self.sleep_time)
					self.stop()
			self.consecutive_presses.clear()
			return
		
		pressed_counter=0
		released_counter=0
		new_history_of_keys=list(set(self.history_of_keys))
		if len(new_history_of_keys)>=1:
			first_ever=new_history_of_keys[0]
			for i in new_history_of_keys:
				if i.split(" ")[1]=="pressed":
					pressed_counter+=1
				elif i.split(" ")[1]=="released":
					released_counter+=1
					
				if pressed_counter==released_counter:
					print("clearing history")
					self.history_of_keys.clear()
					break
					
				if i.split(" ")[0]==first_ever.split(" ")[0] and i.split(" ")[1]!=first_ever.split(" ")[1]:
					pass
			
			
		if direction in self.acceptable_dirs:
			for key,value in self.dirs_and_functions.items():
				if direction==key:
					value()




	def drive(self):
		GPIO.output(self.go_sleep,True)
		
		GPIO.output(self.ma_en,True)
		GPIO.output(self.ma_dir,True)
		
		GPIO.output(self.mb_en,True)
		GPIO.output(self.mb_dir, False)
		
		
		

	def reverse(self):
		GPIO.output(self.go_sleep,True)
		
		GPIO.output(self.ma_en,True)
		GPIO.output(self.ma_dir,False)
		
		GPIO.output(self.mb_en,True)
		GPIO.output(self.mb_dir, True)
		
		
	def left(self):
		GPIO.output(self.go_sleep,True)
		
		GPIO.output(self.ma_en,True)
		GPIO.output(self.ma_dir,False)
		
		GPIO.output(self.mb_en,True)
		GPIO.output(self.mb_dir, False)
		
		
	def right(self):
		GPIO.output(self.go_sleep,True)
		
		GPIO.output(self.ma_en,True)
		GPIO.output(self.ma_dir,True)
		
		GPIO.output(self.mb_en,True)
		GPIO.output(self.mb_dir, True)
		
	def turn(self, ma_pwm, mb_pwm, dir_a, dir_b):
		
		GPIO.output(self.go_sleep,True)
		
		self.ma_en_pwm.ChangeDutyCycle(ma_pwm)
		GPIO.output(self.ma_dir,dir_a)
		
		self.mb_en_pwm.ChangeDutyCycle(mb_pwm)
		GPIO.output(self.mb_dir, dir_b)
		
	def stop(self):
		for i in self.motor_outputs:
			GPIO.output(i,False)
		


if __name__ == '__main__':
	print("All good")

	test = ControlCar()
	test.turn(100,100, False, False)
	time.sleep(2)
	test.turn(0, 0, True, False)
	test.stop()
