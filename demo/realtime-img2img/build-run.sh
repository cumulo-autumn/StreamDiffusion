#!/bin/bash
cd frontend
npm install
npm run build
if [ $? -eq 0 ]; then
    echo -e "\033[1;32m\nfrontend build success \033[0m"
else
    echo -e "\033[1;31m\nfrontend build failed\n\033[0m" >&2  exit 1
fi
cd ../
#check if var PIPELINE is set otherwise get default
if [ -z ${PIPELINE+x} ]; then
    PIPELINE="controlnet"
fi
if [ -z ${COMPILE+x} ]; then
    COMPILE="--sfast"
fi
echo -e "\033[1;32m\npipeline: $PIPELINE \033[0m"
echo -e "\033[1;32m\ncompile: $COMPILE \033[0m"
python3 run.py --port 7860 --host 0.0.0.0 --pipeline $PIPELINE $COMPILE
