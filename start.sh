#!/bin/bash
pip install -r requirements.txt
uvicorn App.server:app --host 0.0.0.0 --port $PORT

