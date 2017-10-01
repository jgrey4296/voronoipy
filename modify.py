import numpy as np
from cairo_utils.dcel import DCEL
from cairo_utils.dcel.constants import FaceE, EdgeE, VertE
from numpy.random import choice, sample, random
from os.path import isfile, join, exists
from voronoipy import voronoi
import IPython
import cairo_utils as utils
import pickle
import random as rnd
from itertools import cycle

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
    chosen_face = choice(list(faces.values()))
    #chosen_face = faces[384]
    logging.info("Chosen face: {}".format(chosen_face.index))
    # for face in dcel.faces:
    #     face.data[FaceE.NULL] = True

    #del chosen_face.data[FaceE.NULL]

    for edge in dcel.halfEdges:
        edge.data[EdgeE.NULL] = True

    colours = [
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 1, 0, 0],
        [0, 0.5, 0.5, 0],
        [0, 0, 1, 0],
        [0.5, 0, 0.5, 0]
        ]

        
    best = find_closest_faces(dcel, n=10)
    totalEdges = 0

    interior, exterior = get_edges_for_faces(dcel, best)
    alreadyDealtWith = set()
    for edge in interior:
        assert(edge in dcel.halfEdges)
        if EdgeE.NULL in edge.data:
            del edge.data[EdgeE.NULL]
        edge.data[EdgeE.COLOUR] = [0, 1, 0, 1]
    for edge in exterior:
        assert(edge in dcel.halfEdges)
        if EdgeE.NULL in edge.data:
            del edge.data[EdgeE.NULL]
        edge.data[EdgeE.COLOUR] = [0, 0.5, 0.5, 1]

    for face in best:
        face.data[FaceE.FILL] = True
        
        

        
    # face = choice(dcel.faces, 1)[0]
    # c = random(4)
    # face.data[FaceE.FILL] = c
    # face.data[FaceE.TEXT] = face.index
    # del face.data[FaceE.NULL]

    # for e in face.outerBoundaryEdges:
    #     if e.face is not None:
    #         if FaceE.NULL in e.face.data:
    #             del e.face.data[FaceE.NULL]
    #         c = random(4)
    #         e.face.data[FaceE.FILL] = c

    # face2 = choice(face.outerBoundaryEdges, 1)[0].face
    # for e in face2.outerBoundaryEdges:
    #     if e.face is not None:
    #         if FaceE.NULL in e.face.data:
    #             del e.face.data[FaceE.NULL]
    #         c = random(4)
    #         e.face.data[FaceE.FILL] = c

    # dcel.constrain_to_circle(np.array((0.5, 0.5)), 0.45)
            
    # for edge in dcel.halfEdges:
    #     edge.data[EdgeE.NULL] = True
    
    # for vert in dcel.vertices:
    #     vert.data[VertE.NULL] = True


def find_closest_faces(dcel, point=np.array((0.5, 0.5)), n=5):
    """ Return the N closest faces to a specified position """
    logging.info("Finding closest faces")
    assert(isinstance(dcel, DCEL))
    assert(isinstance(point, np.ndarray))
    assert(isinstance(n, int))
    distances = [utils.math.get_distance(x.site, point) for x in dcel.faces]
    paired = list(zip(distances, dcel.faces))
    paired.sort()
    best = paired[:n]
    return [x[1] for x in best]

def get_edges_for_faces(dcel, faces):
    """ Return a tuple of (interior, exterior) edges given a group of faces """
    logging.info("Getting edges for faces")
    assert(isinstance(dcel, DCEL))
    assert(isinstance(faces, list))
    assert(all([isinstance(x, utils.dcel.Face) for x in faces]))
    faceSet = set(faces)
    allEdges = set([e for x in faces for e in x.edgeList] + [e for x in faces for e in x.outerBoundaryEdges])
    interior_edges = [e for e in allEdges if e.face in faceSet and e.twin.face in faceSet]
    exterior_edges = [e for e in allEdges if e.face in faceSet and e.twin.face not in faceSet]
    return (interior_edges, exterior_edges)



if __name__ == "__main__":
    the_dcel = DCEL.loadfile(DCEL_PICKLE)
    utils.drawing.clear_canvas(ctx)
    modify(the_dcel)
    
    utils.dcel.drawing.drawDCEL(ctx, the_dcel, edges=True)
    utils.drawing.write_to_png(surface, save_string)
