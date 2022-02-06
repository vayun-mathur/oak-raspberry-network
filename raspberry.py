import sys
import socket
import threading
from random import random
from time import sleep

from flask import Flask, render_template, Response, url_for, copy_current_request_context
from flask_socketio import SocketIO, emit
from camera import Camera
from data_stream import DataStream


app = Flask(__name__)

socketio = SocketIO(app, async_mode=None, logger=False, engineio_logger=False)

HOST = '' 
PORT = 12801

data_stream = DataStream(PORT)

c = Camera(data_stream)

thread = threading.Thread()
thread_stop_event = threading.Event()

def randomNumberGenerator():
    """2
    Generate a random number every 1 second and emit to a socketio instance (broadcast)
    Ideally to be run in a separate thread?
    """
    #infinite loop of magical random numbers
    print("Making random numbers")
    gen = data_stream.read()
    while not thread_stop_event.isSet():
        if data_stream.connected():
            data = next(gen)
            socketio.emit('newdata', {'data': data}, namespace='/test')
        socketio.sleep(0.01)
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(c.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    #Start the random number generator thread only if the thread has not been started before.
    if not thread.is_alive():
        print("Starting Thread")
        thread = socketio.start_background_task(randomNumberGenerator)

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

socketio.run(app)
