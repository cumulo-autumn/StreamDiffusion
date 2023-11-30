import io
import socket
from typing import *


class UDP:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __del__(self):
        self.sock.close()

    def send_udp_data(self, images):
        img_byte_arr = io.BytesIO()
        images.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()
        self.sock.sendto(img_byte_arr, (self.ip, self.port))


def receive_udp_data(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    data, addr = sock.recvfrom(65535)  # 65535 is the maximum UDP packet size
    sock.close()
    return data
