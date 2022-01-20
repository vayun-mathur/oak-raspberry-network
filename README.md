# OAK Raspberry Network

This project was created to stream data from the OpenCV OAK-D camera and compute module
 over a network connection using a raspberry pi.

## Dependencies
### Raspberry Pi
```
pip install -r requirements.txt
```

### Client computer
A Java runtime at least Java 11 needs to be installed on the client computer to run the
java program to recieve the information.

## Files

`models/`, `templates/`, and `camera_detection.py` must be placed on the raspberry pi.

`python.java` must be run on the client device (the one recieving the data 
from the OAK and Raspberry Pi)