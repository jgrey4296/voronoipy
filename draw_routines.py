import math
import cairo
from cairo import OPERATOR_SOURCE
import numpy as np
from numpy import pi
from numpy import linspace
from numpy import cos
from numpy import sin
from numpy.random import random
from scipy.interpolate import splprep
from scipy.interpolate import splev
import utils
import IPython
import logging
from random import choice
#Drawing classes
from voronoi import Voronoi

#DEBUG Stages and amounts
DRAW_INTERMEDIATE = False
PRE_DRAW_EDGES = False
PRE_DRAW_FACES = False
DRAW_EDGES = False
DRAW_FACES = True
FACE_DRAW_LIMIT = 12
EDGE_DRAW_LIMIT = 100
FACE_CENTRING = True

#Globals and constants:
PIX = 1/pow(2,10)
op = None
cairo_surface = None
cairo_context = None
filename = None
numOfElements = 10
iterationNum = 10
branchIterations = 100
voronoi_nodes = 20

#Processing types:
granulate = True
interpolateGranules = False
interpolate = True
interpolateGrains = False

#instances of drawing classes
drawInstance = None
branchInstance = None
voronoiInstance = None
vexpInstance = None

#top level draw command:
def draw(ctx,X_size,Y_size,surface=None,filenamebase="cairo_render"):
    """ The generic setup function main calls for all drawing """
    logging.info("Drawing: {}".format(drawOption))
    #modify and update globals:
    global op
    global drawInstance
    global branchInstance
    global voronoiInstance
    global vexpInstance
    global cairo_surface
    global cairo_context
    global filename
    cairo_surface = surface
    cairo_context = ctx
    filename = filenamebase
    op = ctx.get_operator()
    voronoiInstance = Voronoi(ctx,(X_size,Y_size),voronoi_nodes)
    utils.clear_canvas(ctx)
    
    drawVoronoi(X_size,Y_size)
    
    if surface:
        logging.info("Writing final image")
        utils.write_to_png(surface,filenamebase)
    
#----------
# VORONOI Generation:
#----------
def drawVoronoi(X_size,Y_size):
    """ Step through the construction of a voronoi diagram """
    i = 0
    result = True
    siteLocations = None
    loaded = False
    try: #try to load a saved set of points
        siteLocations = voronoiInstance.load_graph()
        loaded = True
    except Exception as e:
        logging.warn("Using Default Locations")
    #setup the points internally
    siteLocations = voronoiInstance.initGraph(data=siteLocations)
    #Save the points for reuse
    if not loaded:
        voronoiInstance.save_graph(siteLocations)
        
    #Draw the starting set of points    
    voronoiInstance.draw_intermediate_states()
    utils.write_to_png(cairo_surface,filename,i)
    
    while result:
        i += 1
        logging.info("\n---------- Calculating step {}".format(i))
        logging.info(voronoiInstance.beachline)
        
        #The main algorithm step:
        result = voronoiInstance.calculate(i)
        
        logging.info("----- Beachline Modifications:")
        logging.info(voronoiInstance.beachline)
        logging.info("----")

        #Debug intermediate images:
        if DRAW_INTERMEDIATE:
            voronoiInstance.draw_intermediate_states()
            utils.write_to_png(cairo_surface,filename,i)

        #rough infinite loop guard
        if i > 150:
            raise Exception('Voronoi has run too long')

    logging.info("-------------------- Finalising")

    if PRE_DRAW_EDGES:
        logging.info("DRAWING PRE_EDGES")
        for e in voronoiInstance.dcel.halfEdges:
            if e.index > EDGE_DRAW_LIMIT:
                break;
            if e.drawn:
                continue
            logging.info("Drawing edge: {}".format(e.index))
            edgeName = "{}_edge_pre_{}".format(filename,e.index)
            utils.draw_dcel_halfEdge(voronoiInstance.ctx,e)
            utils.write_to_png(cairo_surface,edgeName)
            e.drawn = True
            #e.twin.drawn = True

    #calculations finished, Do the first half of finalising
    #complete, then purge, edges
    dcel = voronoiInstance.finalise_DCEL(surface=cairo_surface,filename=filename)

    #debug edges that fail verification    
    troublesomeEdges = dcel.verify_edges()
    utils.clear_canvas(voronoiInstance.ctx)
    if len(troublesomeEdges) > 0:
        for e in troublesomeEdges:
            utils.draw_dcel_halfEdge(voronoiInstance.ctx,e,clear=False)
        utils.write_to_png(cairo_surface,"{}_TROUBLESOME_EDGES".format(filename))
        IPython.embed()

    #Draw each face individually
    if PRE_DRAW_FACES:
        logging.info("DRAWING FACES")
        for f in dcel.faces:
            if f.index > FACE_DRAW_LIMIT:
                break;
            logging.info("Drawing face: {}".format(f.index))
            faceName = "{}_face_{}".format(filename,f.index)
            utils.draw_dcel_single_face(voronoiInstance.ctx,dcel,f,clear=True,force_centre=FACE_CENTRING)
            utils.write_to_png(cairo_surface,faceName)

    dcel = voronoiInstance.complete_faces()

    if DRAW_EDGES:
        logging.info("DRAWING POST EDGES")
        for e in dcel.halfEdges:
            if e.index > EDGE_DRAW_LIMIT:
                break;
            #if e.drawn:
            #    continue
            logging.info("Drawing edge: {}".format(e.index))
            edgeName = "{}_edge_{}".format(filename,e.index)
            utils.draw_dcel_halfEdge(voronoiInstance.ctx,e)
            utils.write_to_png(cairo_surface,edgeName)
            e.drawn = True
            #e.twin.drawn = True
    
    #Draw each face individually
    if DRAW_FACES:
        logging.info("DRAWING POST-FACES")
        for f in dcel.faces:
            if f.index > EDGE_DRAW_LIMIT:
                break;
            logging.info("Drawing face post completion: {}".format(f.index))
            faceName = "{}_face_pc_{}".format(filename,f.index)
            utils.draw_dcel_single_face(voronoiInstance.ctx,dcel,f,clear=True,force_centre=FACE_CENTRING)
            utils.write_to_png(cairo_surface,faceName)
    

    voronoiInstance.draw_voronoi_diagram()
    finalFilename = "{}-FINAL".format(filename)
    utils.write_to_png(cairo_surface,finalFilename,i+1)
