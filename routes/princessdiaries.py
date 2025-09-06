from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
import math

from flask import request
from routes import app


@app.route('/princessdiaries', methods=['POTS'])
def princessdiaries():
    return 'Python Template'