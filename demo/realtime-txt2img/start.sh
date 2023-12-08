pip install -r requirements.txt
cd view
pnpm build
pnpm preview &
cd ../server
python3 main.py
