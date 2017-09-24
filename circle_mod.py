from cairo_utils.dcel import DCEL
from cairo_utils.dcel.constants import FaceE, EdgeE, VertE
from numpy.random import choice, sample, random
from os.path import isfile, join, exists
from voronoipy import voronoi
import IPython
import cairo_utils as utils
import numpy as np
import pickle
import random as rnd
from cairo_utils.dcel.Line import Line


DCEL_PICKLE = "dcel.pkl"
save_string_unmod = join("imgs", "circle_unmod_test")
save_string = join("imgs", "circle_mod_test")
save_string_faces = join("imgs", "circle_face_test")
N = 10

# Setup root_logger:
import logging as root_logger
LOGLEVEL = root_logger.DEBUG
LOG_FILE_NAME = "log.circle"
root_logger.basicConfig(filename=LOG_FILE_NAME, level=LOGLEVEL, filemode='w')

console = root_logger.StreamHandler()
console.setLevel(root_logger.INFO)
root_logger.getLogger('').addHandler(console)
logging = root_logger.getLogger(__name__)
##############################

surface, ctx, size, N = utils.drawing.setup_cairo(N=N, font_size=0.015)
RADIUS = 0.45
CENTRE = np.array([0.5, 0.5])

def modify(dcel):
    constrain_half_edges_to_circle(dcel)
                

def constrain_half_edges_to_circle(dcel, radius=RADIUS, centre=CENTRE, replace_with_arcs=False):
    removed_edges = []
    modified_edges = []
    
    for he in dcel.halfEdges:
        results = he.within_circle(centre, radius)
        arr = he.origin.toArray()
        # if arr[1] > 0.6 and random() < 0.35:
        #     he.data[EdgeE.TEXT] = True
        if he.index == 2070:
            he.data[EdgeE.TEXT] = True
        if all(results): #if both within circle: leave
            continue
        elif not any(results): #if both without: remove
            he.markForCleanup()
            removed_edges.append(he)
        else: #one within, one without, modify
            #Get the further point
            closer, further, isOrigin = he.getCloserAndFurther(centre, radius)
            if isOrigin:
                asLine = Line.newLine(he.origin, he.twin.origin, np.array([0,0,1,1]))
            else:
                asLine = Line.newLine(he.twin.origin, he.origin, np.array([0,0,1,1]))
            intersection = asLine.intersect_with_circle(centre, radius)
            if intersection[0] is None:
                closest = intersection[1]
            elif intersection[1] is None:
                closest = intersection[0]
            else:
                closest = intersection[np.argmin(utils.get_distance(np.array(intersection), further))]
            newVert = dcel.newVertex(*closest)
            orig1, orig2 = he.getVertices()
            he.clearVertices()
            #move it onto the circle
            if isOrigin:
                #origin is closer, replace the twin
                he.addVertex(orig1)
                he.addVertex(newVert)
            else:
                #twin is closer, replace the origin
                he.addVertex(newVert)
                he.addVertex(orig2)
            modified_edges.append(he)

    #todo: fixup faces
            
    dcel.purge_edges()
    dcel.purge_vertices()
    dcel.purge_faces()
    dcel.purge_infinite_edges()
    dcel.complete_faces()
    

    
if __name__ == "__main__":
    the_dcel = DCEL.loadfile(DCEL_PICKLE)
    the_dcel.verify_faces_and_edges()
    #Pre MOD
    utils.clear_canvas(ctx, colour=[0,0,0,1])
    utils.drawDCEL(ctx, the_dcel, edges=True, faces=False)
    ctx.set_source_rgba(0,1,0,1)
    ctx.set_line_width(0.002)
    utils.drawing.drawCircle(ctx, 0.5, 0.5, RADIUS, fill=False)
    utils.write_to_png(surface, save_string_unmod)
    #Post MOD
    utils.clear_canvas(ctx, colour=[0,0,0,1])
    utils.drawDCEL(ctx, the_dcel, edges=False, faces=True, verts=True)
    modify(the_dcel)
    utils.drawDCEL(ctx, the_dcel, edges=True, faces=False, text=False)
    ctx.set_source_rgba(0,1,0,1)
    ctx.set_line_width(0.002)
    utils.drawing.drawCircle(ctx, 0.5, 0.5, RADIUS, fill=False)
    utils.write_to_png(surface, save_string)

    #finalised faces only:
    utils.clear_canvas(ctx)
    ctx.set_source_rgba(0,1,0,1)
    ctx.set_line_width(0.002)
    #utils.drawing.drawCircle(ctx, 0.5, 0.5, RADIUS, fill=False)
    utils.drawDCEL(ctx, the_dcel, faces=True)
    utils.write_to_png(surface, save_string_faces)
    
