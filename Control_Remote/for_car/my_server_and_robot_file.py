import socket
import threading,sys
import Actuation
import os,time
sys.path.append("../")
sys.path.append("/home/pi/pi-camera-stream-flask")
#import main_cam
import mutual
import detect
import object_detect
import workin_object_ident

class Server:
	def __init__(self):
		self.HEADER=mutual.header
		self.PORT=mutual.port
		self.SERVER=self.get_ip_address()
		self.ADDR=(self.SERVER,self.PORT)
		self.FORMAT="utf-8"
		self.LAST_MESSAGE=""

		self.IS_CONNECTED=True

		self.server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.server.bind(self.ADDR)
		self.my_car=Actuation.ControlCar()

	def handle_client(self,conn,addr):
		print("{} connected".format(addr))

		self.IS_CONNECTED=True
		while self.IS_CONNECTED:
			msg_length=conn.recv(self.HEADER).decode(self.FORMAT)
			if msg_length:
				msg_length=int(msg_length)
				msg=conn.recv(msg_length).decode(self.FORMAT)
				
				print("[{}] says [{}]".format(addr,msg))
				
				if msg.lower() in mutual.exit_keywords:
					self.IS_CONNECTED=False
				else:
					self.LAST_MESSAGE=msg
					self.my_car.consecutive_presses.append(self.LAST_MESSAGE)
					self.my_car.car_cpu(self.LAST_MESSAGE)

		conn.close()
		print("Client disconnected...")
		self.my_car.stop()



	def start(self):
		print("Server listening on {}".format(self.SERVER))
		self.server.listen()

		while True:
			conn,addr=self.server.accept()
			thread=threading.Thread(target=self.handle_client,args=(conn,addr,))
			thread.start()
			print("Total connections: {}".format(threading.activeCount()-1))
		
	def get_ip_address(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(("8.8.8.8", 80))
		ip_add=s.getsockname()[0]
		s.close()
		return ip_add

if __name__=="__main__":
	print("All good")

	server_instance=Server()

	t1=threading.Thread(target=server_instance.start)
	t1.start()
	print("t1 started")
	time.sleep(0.5)

	t2=threading.Thread(target=workin_object_ident.main)
	t2.start()
	print("t2 started")
