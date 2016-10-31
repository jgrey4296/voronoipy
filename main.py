#!/users/jgrey/anaconda/bin/python
import sys
import time
import math
import cairo
import draw_routines
import logging
import voronoi
import utils

#constants:
#Size of the screen:
N = 12
X = pow(2,N)
Y = pow(2,N)
imgPath = "./imgs/"
imgName = "initialTest"
currentTime = time.gmtime()
FONT_SIZE = 0.03
VORONOI_SIZE = 200
#format the name of the image to be saved thusly:
saveString = "%s%s_%s-%s-%s_%s-%s" % (imgPath,
                                          imgName,
                                          currentTime.tm_min,
                                          currentTime.tm_hour,
                                          currentTime.tm_mday,
                                          currentTime.tm_mon,
                                          currentTime.tm_year)

#setup logging:
LOGLEVEL = logging.DEBUG
logFileName = "voronoi.log"
logging.basicConfig(filename=logFileName,level=LOGLEVEL,filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

#setup
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
ctx = cairo.Context(surface)
ctx.scale(X,Y)
ctx.set_font_size(FONT_SIZE)

#Setup the voronoi diagram:
voronoiInstance = voronoi.Voronoi(ctx,(X,Y),num_of_nodes=VORONOI_SIZE)
voronoiInstance.initGraph()
voronoiInstance.calculate_to_completion()

dcel = voronoiInstance.finalise_DCEL()

#repeatedly relax 

#Manipulate DCEL to create map

#Draw 
utils.drawDCEL(ctx,dcel)
utils.write_to_png(surface,saveString)
