import sys
import time
import math
import cairo
import logging
import voronoi
import utils
import IPython
import pickle
from os.path import isfile
import dcel
import voronoi_map

#setup logging
LOGLEVEL = logging.DEBUG
logFileName = "mapGen.log"
logging.basicConfig(filename=logFileName,level=LOGLEVEL,filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

#setup cairo
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
ctx = cairo.Context(surface)
ctx.scale(X,Y)
ctx.set_font_size(FONT_SIZE)


#DCEL Loading:
DCEL_PICKLE = "dcel.pkl"
if not isfile(DCEL_PICKLE):
    raise Exception("DCEL pickle doesnt exist")

with open(DCEL_PICKLE,'rb') as f:
    the_dcel = pickle.load(f)
    the_dcel.calculate_quad_tree()

if not isinstance(the_dcel,dcel.DCEL):
    raise Exception("Bad DCEL Load")

#dcel loaded, transform into a map:
vmap = voronoi_map.Map(the_dcel)



#save it and draw?
