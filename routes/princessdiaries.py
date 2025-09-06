from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
import math

from flask import request
from routes import app


@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'