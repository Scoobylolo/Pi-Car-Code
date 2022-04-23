import socket
import sys,os
import threading
sys.path.append("../")
import mutual
sys.path.append("../../pi_ip_cam")
import main_cam_file


if os.name=="nt": #windows
	from pynput.keyboard import Key, Listener
elif os.name=="posix": #linux
	from curtsies import Input


class Client:
	def __init__(self):
		self.HEADER=mutual.header
		self.PORT=mutual.port
		self.SERVER=socket.gethostbyname(socket.gethostname())
		self.ADDR=(self.SERVER,self.PORT)
		self.FORMAT="utf-8"

		self.client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

		self.client.connect(("10.0.0.118",self.PORT))

	def get_ip_address(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(("8.8.8.8", 80))
		ip_add=s.getsockname()[0]
		s.close()
		return ip_add

	def send(self,msg):
		message=msg.encode(self.FORMAT)
		msg_length=len(message)
		send_length=str(msg_length).encode(self.FORMAT)
		send_length+=b" " * (self.HEADER-len(send_length))
		self.client.send(send_length)
		self.client.send(message)

	def get_key(self):
		print("Press esc to exit")
		print("\n{} - move forward\n{} - move backward\n{} - move left\n{} - move right\n".
		format(mutual.forward,mutual.back,mutual.left,mutual.right))

		if os.name=="nt": #windows

			def on_press(key):
				# print('{0} pressed'.format(key))
				self.send(str(key).replace("'",""))
				if key == Key.esc:
					print("Quitting keylogger...")
					return False
		

			# Collect events until released
			with Listener(on_press=on_press) as listener:
				listener.join()

		elif os.name=="posix": #linux
			with Input(keynames='curses') as IG:
				for i in IG:
					the_key=repr(i).replace("'","")
					# print(the_key)

					self.send(the_key)

					if the_key=="\\x1b": #this is escape
						print("Quitting keylogger...")
						break


if __name__=="__main__":
	print("All good")
	print("Starting client...")

	client_instance=Client()

	client_thread=threading.Thread(target=client_instance.get_key)
	client_thread.start()

	camera_thread=threading.Thread(target=main_cam_file.app.run,
	kwargs={"host":"0.0.0.0", "debug":False})
	camera_thread.start()

