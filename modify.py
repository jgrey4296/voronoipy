from cairo_utils.dcel import DCEL
from cairo_utils.dcel.constants import FaceE, EdgeE, VertE
from numpy.random import choice, sample, random
from os.path import isfile, join, exists
from voronoipy import voronoi
import IPython
import cairo_utils as utils
import pickle
import random as rnd

DCEL_PICKLE = "dcel.pkl"
save_string = join("imgs", "v_modified")
N = 10

# Setup root_logger:
import logging as root_logger
LOGLEVEL = root_logger.DEBUG
LOG_FILE_NAME = "log.modify"
root_logger.basicConfig(filename=LOG_FILE_NAME, level=LOGLEVEL, filemode='w')

console = root_logger.StreamHandler()
console.setLevel(root_logger.INFO)
root_logger.getLogger('').addHandler(console)
logging = root_logger.getLogger(__name__)
##############################

surface, ctx, size, N = utils.drawing.setup_cairo(N=N)

def modify(dcel):
    faces = {x.index : x for x in dcel.faces}
    for face in dcel.faces:
        face.data[FaceE.NULL] = True

    face = choice(dcel.faces, 1)[0]
    c = random(4)
    face.data[FaceE.FILL] = c
    face.data[FaceE.TEXT] = face.index
    del face.data[FaceE.NULL]

    for e in face.innerComponents:
        if e.face is not None:
            if FaceE.NULL in e.face.data:
                del e.face.data[FaceE.NULL]
            c = random(4)
            e.face.data[FaceE.FILL] = c

    face2 = choice(face.innerComponents, 1)[0].face
    for e in face2.innerComponents:
        if e.face is not None:
            if FaceE.NULL in e.face.data:
                del e.face.data[FaceE.NULL]
            c = random(4)
            e.face.data[FaceE.FILL] = c
                
    # for edge in dcel.halfEdges:
    #     edge.data[EdgeE.NULL] = True
    
    # for vert in dcel.vertices:
    #     vert.data[VertE.NULL] = True
                  

if __name__ == "__main__":
    the_dcel = DCEL.loadfile(DCEL_PICKLE)
    modify(the_dcel)
    utils.clear_canvas(ctx)
    utils.drawDCEL(ctx, the_dcel, edges=True)
    utils.write_to_png(surface, save_string)
