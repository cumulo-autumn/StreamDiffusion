import io
import socket

from PIL import Image


class UDP:
    def __init__(self, ip: str, port: int, max_size: int = 65507):
        self.ip = ip
        self.port = port
        self.max_size = max_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __del__(self):
        self.sock.close()

    def send_udp_data(self, image: Image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        if len(img_byte_arr) <= self.max_size:
            self.sock.sendto(img_byte_arr, (self.ip, self.port))
        else:
            print("Warning: Image size is too large. Image is not sent.")


def receive_udp_data(ip: str, port: int) -> bytes:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    data, addr = sock.recvfrom(65535)
    sock.close()
    return data
