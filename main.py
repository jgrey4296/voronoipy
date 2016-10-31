#!/users/jgrey/anaconda/bin/python

import sys
import time
import math
import cairo
import draw_routines
import logging

#constants
N = 12
X = pow(2,N)
Y = pow(2,N)
imgPath = "./imgs/"
imgName = "initialTest"
currentTime = time.gmtime()
FONT_SIZE = 0.03
#format the name of the image to be saved thusly:
saveString = "%s%s_%s-%s-%s_%s-%s" % (imgPath,
                                          imgName,
                                          currentTime.tm_min,
                                          currentTime.tm_hour,
                                          currentTime.tm_mday,
                                          currentTime.tm_mon,
                                          currentTime.tm_year)


#get the type of drawing to do from the command line argument:
if len(sys.argv) > 1:
    drawRoutineName = sys.argv[1]
else:
    drawRoutineName = "circles"

#setup logging:
LOGLEVEL = logging.DEBUG
logFileName = drawRoutineName + ".log"
logging.basicConfig(filename=logFileName,level=LOGLEVEL,filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

#setup
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
ctx = cairo.Context(surface)
ctx.scale(X,Y)
ctx.set_font_size(FONT_SIZE)


#Drawing:
draw_routines.draw(ctx,drawRoutineName,X,Y,surface=surface,filenamebase=saveString)
    
# #write to file: - DEPRECATED: moved into draw routines
# print('Saving')
# surface.write_to_png (saveString)

