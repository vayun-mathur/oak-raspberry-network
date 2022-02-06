import socket
import threading

def find_connection(data_stream):
    data_stream.s.listen(1)
    conn, addr = data_stream.s.accept()
    data_stream.conns.append(conn)

class DataStream:
    def __init__(this, port):
        this.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        this.s.bind(('', port))
        this.conns = []
        threading.Thread(target=find_connection, args=[this]).start()
    
    def write(this, string):
        for conn in this.conns:
            conn.sendall(string.encode('utf-8'))
