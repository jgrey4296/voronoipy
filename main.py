import sys
import time
import math
import cairo
import logging
import IPython
from os.path import isfile,join,exists
import random
from numpy.random import choice

import pickle
from voronoipy import voronoi
import cairo_utils as utils
from cairo_utils.dcel import DCEL


#constants:
#Size of the screen:
imgPath = "./imgs"
imgName = "initialTest"
DCEL_PICKLE = "dcel.pkl"
currentTime = time.gmtime()
FONT_SIZE = 0.03
VORONOI_SIZE = 100
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
logFileName = "log.voronoi"
logging.basicConfig(filename=logFileName,level=LOGLEVEL,filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

#setup
surface, ctx, size = utils.drawing.setup_cairo(size_power=N)

#--------------------------------------------------------------------------------

def generate_voronoi():
    """ Generate and relax a voronoi diagram, then pickle the resulting DCEL  """
    #Setup the voronoi diagram:
    logging.info("Creating Initial Voronoi Diagram")
    voronoiInstance = voronoi.Voronoi(ctx,(size, size),num_of_nodes=VORONOI_SIZE)
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
        pickle.dump(the_dcel.export_data(),f)

#--------------------------------------------------------------------------------

def load_file_and_average():
    logging.info("Opening DCEL pickle")
    the_dcel = DCEL()
    with open(DCEL_PICKLE,'rb') as f:
        dcel_data = pickle.load(f)
    the_dcel.import_data(dcel_data)

    #Select a number of faces to fill:
    NUM_OF_FACES = 10
    aface = choice(the_dcel.faces, NUM_OF_FACES)

    for x in aface:
        x.data = {'fill' : True }
    
    #Draw
    logging.info("Drawing final diagram")
    utils.clear_canvas(ctx)
    utils.drawDCEL(ctx,the_dcel)
    utils.write_to_png(surface,"{}__FINAL".format(savePath))
#--------------------------------------------------------------------------------

if __name__ == "__main__":
    if not isfile(DCEL_PICKLE):
        generate_voronoi()
    load_file_and_average()
   
