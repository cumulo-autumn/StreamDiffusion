from fastapi import FastAPI

from config import args
from device import device, torch_dtype
from app_init import init_app
from user_queue import user_data
from img2img import Pipeline

print("DEVICE:", device)
print("TORCH_DTYPE:", torch_dtype)
args.pretty_print()

app = FastAPI()

pipeline = Pipeline(args, device, torch_dtype)
init_app(app, user_data, args, pipeline)


if __name__ == "__main__":
    import uvicorn
    from config import args

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
