import sys
import time
import math
import cairo
import logging
import IPython
import numpy as np
from os.path import isfile,join,exists
import random
from numpy.random import choice, sample

import pickle
from voronoipy import voronoi
import cairo_utils as utils
from cairo_utils.dcel import DCEL
from cairo_utils.dcel.constants import FaceE


#constants:
#Size of the screen:
imgPath = "./imgs"
imgName = "initialTest"
DCEL_PICKLE = "voronoi"
currentTime = time.gmtime()
VORONOI_SIZE = 20
RELAXATION_ITER = 0
#RELAXATION_AMNT = 0.4
RELAXATION_AMNT = sample
N = 12
VORONOI_DEBUG = True
DRAW_RELAXATIONS = True
DRAW_FINAL = True
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
logFileName = "log.voronoi"
logging.basicConfig(filename=logFileName,level=LOGLEVEL,filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

#setup
surface, ctx, size, N = utils.drawing.setup_cairo(N=N, scale=False, cartesian=True,
                                                  font_size=200)
bbox = np.array([0,0,size,size])
#--------------------------------------------------------------------------------

def generate_voronoi():
    """ Generate and relax a voronoi diagram, then pickle the resulting DCEL  """
    #Setup the voronoi diagram:
    logging.info("Creating Initial Voronoi Diagram")
    voronoiInstance = voronoi.Voronoi((size, size),
                                      num_of_nodes=VORONOI_SIZE,
                                      bbox=bbox,
                                      debug_draw=VORONOI_DEBUG, n=N)
    voronoiInstance.initGraph()
    voronoiInstance.calculate_to_completion()
    #repeatedly relax and rerun
    for i in range(RELAXATION_ITER):
        logging.info("-------------------- Relaxing iteration: {}".format(i))
        assert(voronoiInstance.nodeSize == VORONOI_SIZE)
        dcel = voronoiInstance.finalise_DCEL()
        for f in dcel.faces:
            f.data[FaceE.CENTROID] = True
        if DRAW_RELAXATIONS:
            utils.drawing.clear_canvas(ctx, bbox=bbox)
            utils.dcel.drawing.drawDCEL(ctx,dcel)
            utils.drawing.write_to_png(surface,"{}__relaxed_{}".format(savePath,i))
        if callable(RELAXATION_AMNT):
            voronoiInstance.relax(amnt=RELAXATION_AMNT())
        else:
            voronoiInstance.relax(amnt=RELAXATION_AMNT)

    logging.info("Finalised Voronoi diagram, proceeding")
    the_dcel = voronoiInstance.finalise_DCEL()
    the_dcel.savefile(DCEL_PICKLE)

#--------------------------------------------------------------------------------

def load_file_and_average():
    logging.info("Opening DCEL pickle")
    the_dcel = DCEL.loadfile(DCEL_PICKLE)

    #Select a number of faces to fill:
    if len(the_dcel.faces) > 0:
        NUM_OF_FACES = min(VORONOI_SIZE, 10)
        aface = choice(the_dcel.faces, NUM_OF_FACES)

        for x in aface:
            x.data = {'fill' : True }
    
    #Draw
    if DRAW_FINAL:
        logging.info("Drawing final diagram")
        utils.drawing.clear_canvas(ctx, bbox=bbox)
        utils.dcel.drawing.drawDCEL(ctx,the_dcel)
        utils.drawing.write_to_png(surface,"{}__FINAL".format(savePath))
#--------------------------------------------------------------------------------

if __name__ == "__main__":
    if not isfile("{}.dcel".format(DCEL_PICKLE)):
        generate_voronoi()
    load_file_and_average()
   
