import io
import socket
import struct


class UDP:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.buffer_size = 65507

    def __del__(self):
        self.sock.close()

    def send_udp_data(self, images):
        img_byte_arr = io.BytesIO()
        images.save(img_byte_arr, format="JPEG")
        img_data = img_byte_arr.getvalue()

        self.sock.sendto(struct.pack(">I", len(img_data)), (self.ip, self.port))

        for i in range(0, len(img_data), self.buffer_size):
            self.sock.sendto(img_data[i : i + self.buffer_size], (self.ip, self.port))


def receive_udp_data(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    data_size, _ = sock.recvfrom(4)
    data_size = struct.unpack(">I", data_size)[0]

    received = 0
    packets = []

    while received < data_size:
        data, addr = sock.recvfrom(65535)
        packets.append(data)
        received += len(data)

    sock.close()
    return b"".join(packets)
