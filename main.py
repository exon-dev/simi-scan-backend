# server libraries here
from flask import Flask, request, make_response, abort, jsonify

# ml libraries here
import torchvision.transforms as TT

#miscellaneous
from PIL import Image

app = Flask(__name__)

@app.route('/')
def main():
 return "<h1>SimiScan Flask Server Start!</h1>"