#!/users/jgrey/anaconda/bin/python
import sys
import time
import math
import cairo
import logging
import voronoi
import utils
import IPython
import pickle
from os.path import isfile,join,exists
import random
import dcel
from numpy.random import choice

#constants:
#Size of the screen:
N = 12
X = pow(2,N)
Y = pow(2,N)
imgPath = "./imgs"
imgName = "initialTest"
DCEL_PICKLE = "dcel.pkl"
currentTime = time.gmtime()
FONT_SIZE = 0.03
VORONOI_SIZE = 400
RELAXATION_AMNT = 3
#format the name of the image to be saved thusly:
saveString = "{}_{}-{}-{}_{}-{}".format(  imgName,
                                          currentTime.tm_min,
                                          currentTime.tm_hour,
                                          currentTime.tm_mday,
                                          currentTime.tm_mon,
                                          currentTime.tm_year)
savePath = join(imgPath,saveString)

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

#--------------------------------------------------------------------------------

def generate_voronoi():
    """ Generate and relax a voronoi diagram, then pickle the resulting DCEL  """
    
    #Setup the voronoi diagram:
    logging.info("Creating Initial Voronoi Diagram")
    voronoiInstance = voronoi.Voronoi(ctx,(X,Y),num_of_nodes=VORONOI_SIZE)
    voronoiInstance.initGraph()
    voronoiInstance.calculate_to_completion()

    #repeatedly relax and rerun
    for i in range(RELAXATION_AMNT):
        logging.info("-------------------- Relaxing iteration: {}".format(i))
        dcel = voronoiInstance.finalise_DCEL()
        utils.clear_canvas(ctx)
        utils.drawDCEL(ctx,dcel)
        utils.write_to_png(surface,"{}__relaxed_{}".format(savePath,i))
        voronoiInstance.relax()
        
    logging.info("Finalised Voronoi diagram, proceeding")
    the_dcel = voronoiInstance.finalise_DCEL()
    with open(DCEL_PICKLE,'wb') as f:
        try:
            pickle.dump(the_dcel.export_data(),f)
        except RecursionError as e:
            logging.info("Recursion error")
            IPython.embed()

#--------------------------------------------------------------------------------
if not isfile(DCEL_PICKLE):
    generate_voronoi()

logging.info("Opening DCEL pickle")
the_dcel = dcel.DCEL()
with open(DCEL_PICKLE,'rb') as f:
    dcel_data = pickle.load(f)
the_dcel.import_data(dcel_data)

#Manipulate DCEL to create map
NUM_OF_FACES = 10
aface = choice(the_dcel.faces, NUM_OF_FACES)

for x in aface:
    x.data = {'fill' : True }

#FACE_SELECTION = 27
#aface = list(filter(lambda x: x.index == FACE_SELECTION,the_dcel.faces))
#for x in the_dcel.faces:
#    x.sort_edges()

#the_dcel.faces = aface
    
#aface[0].data = {'fill': True}
logging.info("POINT FOR INSPECTING A FACE")
IPython.embed()

#the_dcel.faces = aface

#Draw
logging.info("Drawing final diagram")
utils.clear_canvas(ctx)
utils.drawDCEL(ctx,the_dcel)
utils.write_to_png(surface,"{}__FINAL".format(savePath))

IPython.embed()
