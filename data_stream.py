import socket
import threading
from time import sleep

def find_connection(data_stream):
    data_stream.s.listen(1)
    conn, addr = data_stream.s.accept()
    data_stream.conn = conn

class DataStream:
    def __init__(this, port):
        this.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        this.s.bind(('', port))
        this.conn = None
        threading.Thread(target=find_connection, args=[this]).start()
        this.buffer = ""
    
    def write(this, string):
        if this.conn is not None:
            this.conn.sendall(string.encode('utf-8'))

    def connected(this):
        return this.conn is not None

    def read(this):
        buffer = this.conn.recv(32).decode('utf-8')
        buffering = True
        while buffering:
            if "\n" in buffer:
                (line, buffer) = buffer.split("\n", 1)
                yield line + "\n"
            else:
                more = this.conn.recv(32).decode('utf-8')
                if not more:
                    sleep(0.01)
                else:
                    buffer += more
        if buffer:
            yield buffer
