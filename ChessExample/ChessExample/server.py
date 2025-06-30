import socket
PORT = 5000
BUFFER_SIZE = 1024
def host_game():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", PORT))
    sock.listen(1)
    conn, _ = sock.accept()
    return conn
def join_game(ip: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, PORT))
    return sock