#!/bin/bash
pip install -r requirements.txt 
cd view && npm install && npm run build && cd ..
cd server && python3 main.py
