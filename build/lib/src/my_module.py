import json
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).absolute().parent.parent)

def my_add(x, y):
    return x+y

def my_mul(x, y):
    return x*y

def my_load():
    with open(ROOT_DIR+'/data/params.json', 'r') as f: 
        data = json.load(f)
    return data