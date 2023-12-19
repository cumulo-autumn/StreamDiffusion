#!/bin/bash
cd view && npm install && npm install -g serve && cd ..
pip install -r requirements.txt
cd view && npm run build && serve -s build -l 3000 &
cd server && python3 main.py
