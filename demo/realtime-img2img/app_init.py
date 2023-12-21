from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import markdown2

import logging
import traceback
from config import Args
from user_queue import UserData
import uuid
import time
from types import SimpleNamespace
from util import pil_to_frame, bytes_to_pil, is_firefox
import asyncio
import os
import time

THROTTLE = 1.0 / 120


def init_app(app: FastAPI, user_data: UserData, args: Args, pipeline):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.websocket("/api/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        user_count = user_data.get_user_count()
        if args.max_queue_size > 0 and user_count >= args.max_queue_size:
            print("Server is full")
            await websocket.send_json({"status": "error", "message": "Server is full"})
            await websocket.close()
            return
        try:
            user_id = uuid.uuid4()
            print(f"New user connected: {user_id}")

            await user_data.create_user(user_id, websocket)
            await websocket.send_json(
                {"status": "connected", "message": "Connected", "userId": str(user_id)}
            )
            await websocket.send_json({"status": "send_frame"})
            await handle_websocket_data(user_id, websocket)
        except WebSocketDisconnect as e:
            logging.error(f"WebSocket Error: {e}, {user_id}")
            traceback.print_exc()
        finally:
            print(f"User disconnected: {user_id}")
            user_data.delete_user(user_id)

    async def handle_websocket_data(user_id: uuid.UUID, websocket: WebSocket):
        if not user_data.check_user(user_id):
            return HTTPException(status_code=404, detail="User not found")
        last_time = time.time()
        try:
            while True:
                data = await websocket.receive_json()
                if data["status"] != "next_frame":
                    asyncio.sleep(THROTTLE)
                    continue

                params = await websocket.receive_json()
                params = pipeline.InputParams(**params)
                info = pipeline.Info()
                params = SimpleNamespace(**params.dict())
                if info.input_mode == "image":
                    image_data = await websocket.receive_bytes()
                    if len(image_data) == 0:
                        await websocket.send_json({"status": "send_frame"})
                        continue
                    params.image = bytes_to_pil(image_data)
                await user_data.update_data(user_id, params)
                await websocket.send_json({"status": "wait"})
                if args.timeout > 0 and time.time() - last_time > args.timeout:
                    await websocket.send_json(
                        {
                            "status": "timeout",
                            "message": "Your session has ended",
                            "userId": user_id,
                        }
                    )
                    await websocket.close()
                    return
                await asyncio.sleep(THROTTLE)

        except Exception as e:
            logging.error(f"Error: {e}")
            traceback.print_exc()

    @app.get("/api/queue")
    async def get_queue_size():
        queue_size = user_data.get_user_count()
        return JSONResponse({"queue_size": queue_size})

    @app.get("/api/stream/{user_id}")
    async def stream(user_id: uuid.UUID, request: Request):
        try:
            print(f"New stream request: {user_id}")

            async def generate():
                websocket = user_data.get_websocket(user_id)
                last_params = SimpleNamespace()
                while True:
                    last_time = time.time()
                    params = await user_data.get_latest_data(user_id)
                    if not vars(params) or params.__dict__ == last_params.__dict__:
                        await websocket.send_json({"status": "send_frame"})
                        continue

                    last_params = params
                    image = pipeline.predict(params)

                    if image is None:
                        await websocket.send_json({"status": "send_frame"})
                        continue
                    frame = pil_to_frame(image)
                    yield frame
                    # https://bugs.chromium.org/p/chromium/issues/detail?id=1250396
                    if not is_firefox(request.headers["user-agent"]):
                        yield frame
                    await websocket.send_json({"status": "send_frame"})
                    if args.debug:
                        print(f"Time taken: {time.time() - last_time}")

            return StreamingResponse(
                generate(),
                media_type="multipart/x-mixed-replace;boundary=frame",
                headers={"Cache-Control": "no-cache"},
            )
        except Exception as e:
            logging.error(f"Streaming Error: {e}, {user_id} ")
            traceback.print_exc()
            return HTTPException(status_code=404, detail="User not found")

    # route to setup frontend
    @app.get("/api/settings")
    async def settings():
        info_schema = pipeline.Info.schema()
        info = pipeline.Info()
        if info.page_content:
            page_content = markdown2.markdown(info.page_content)

        input_params = pipeline.InputParams.schema()
        return JSONResponse(
            {
                "info": info_schema,
                "input_params": input_params,
                "max_queue_size": args.max_queue_size,
                "page_content": page_content if info.page_content else "",
            }
        )

    if not os.path.exists("public"):
        os.makedirs("public")

    app.mount("/", StaticFiles(directory="public", html=True), name="public")
