from typing import Dict
from uuid import UUID
import asyncio
from fastapi import WebSocket
from types import SimpleNamespace
from typing import Dict
from typing import Union

UserDataContent = Dict[UUID, Dict[str, Union[WebSocket, asyncio.Queue]]]


class UserData:
    def __init__(self):
        self.data_content: Dict[UUID, UserDataContent] = {}

    async def create_user(self, user_id: UUID, websocket: WebSocket):
        self.data_content[user_id] = {
            "websocket": websocket,
            "queue": asyncio.Queue(),
        }
        await asyncio.sleep(1)

    def check_user(self, user_id: UUID) -> bool:
        return user_id in self.data_content

    async def update_data(self, user_id: UUID, new_data: SimpleNamespace):
        user_session = self.data_content[user_id]
        queue = user_session["queue"]
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
        await queue.put(new_data)

    async def get_latest_data(self, user_id: UUID) -> SimpleNamespace:
        user_session = self.data_content[user_id]
        queue = user_session["queue"]

        try:
            return await queue.get()
        except asyncio.QueueEmpty:
            return None

    def delete_user(self, user_id: UUID):
        user_session = self.data_content[user_id]
        queue = user_session["queue"]
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
        if user_id in self.data_content:
            del self.data_content[user_id]

    def get_user_count(self) -> int:
        return len(self.data_content)

    def get_websocket(self, user_id: UUID) -> WebSocket:
        return self.data_content[user_id]["websocket"]


user_data = UserData()
