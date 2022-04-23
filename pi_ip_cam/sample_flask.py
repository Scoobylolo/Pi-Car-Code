from flask import Flask, render_template, Response, request
# from camera import VideoCamera
import time
import os

app = Flask(__name__)


# self.pi_camera = VideoCamera(flip=False) # flip pi camera if upside down.


@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here

# def gen(self,camera):
#     #get camera frame
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @self.app.route('/video_feed')
# def video_feed():
#     return Response(gen(self.pi_camera),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/lola.html")
def lola():
	return render_template("lola.html")

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False)

	