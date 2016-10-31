import IPython
import math
from math import atan2,sqrt
import logging
import utils
import numpy as np
import pyqtree
import sys

EPSILON = sys.float_info.epsilon
CENTRE = np.array([[0.5,0.5]])
PI = math.pi
TWOPI = 2 * PI
HALFPI = PI * 0.5



# An implementation of a Double-Edge Connected List
# from de Berg's Computational Geometry Book
# Intended for use with cairo

class Vertex:
    """ A Simple vertex for two dimension """

    nextIndex = 0
    
    def __init__(self,x,y,iEdge=None):
        logging.debug("Creating vertex at: {0:.3f} {1:.3f}".format(x,y))
        self.x = x
        self.y = y
        self.incidentEdge = iEdge
        self.halfEdges = []
        self.active = True
        self.index = Vertex.nextIndex
        Vertex.nextIndex += 1

    def eq(self,v2):
        a = self.x - v2.x
        b = self.y - v2.y
        return a <= EPSILON and b <= EPSILON
        
        
    def __str__(self):
        return "({},{})".format(self.x,self.y)
        
    def isEdgeless(self):
        return len(self.halfEdges) == 0
        
    def activate(self):
        self.active = True
        
    def deactivate(self):
        self.active = False
        
    def bbox(self):
        return [self.x-EPSILON,self.y-EPSILON,self.x+EPSILON,self.y+EPSILON]
        
    def registerHalfEdge(self,he):
        self.halfEdges.append(he)
        
    def unregisterHalfEdge(self,he):
        if he in self.halfEdges:
            self.halfEdges.remove(he)
            
            
    def within(self,bbox):
        """ Check the vertex is within [x,y,x2,y2] """
        inXBounds = bbox[0] <= self.x and self.x <= bbox[2]
        inYBounds = bbox[1] <= self.y and self.y <= bbox[3]
        return inXBounds and inYBounds
        
    def outside(self,bbox):
        """ Check the vertex is entirely outside of the bbox [x,y,x2,y2] """
        return not self.within(bbox)

    def toArray(self):
        return np.array([self.x,self.y])
    
class Line:
    """ A line as a start x and y, a direction, and a length """
    
    def __init__(self,sx,sy,dx,dy,l,swapped=False):
        self.source = np.array([sx,sy])
        self.direction = np.array([dx,dy])
        self.length = l
        self.swapped = swapped
        
    def constrain(self,min_x,min_y,max_x,max_y):
        """ Intersect the line with a bounding box, adjusting points as necessary """
        #min and max: [x,y]
        dest = self.destination()
        npArray_line = np.array([*self.source,*dest])
        bbox_lines = [np.array([min_x,min_y,max_x,min_y]),
                      np.array([min_x,max_y,max_x,max_y]),
                      np.array([min_x,min_y,min_x,max_y]),
                      np.array([max_x,min_y,max_x,max_y])]
        #intersect one of the bbox lines
        p = None
        while p is None and len(bbox_lines) > 0:
            p = utils.intersect(npArray_line,bbox_lines.pop())
        if p is not None:
            new_length = sqrt(pow(p[0]-self.source[0],2) + pow(p[1]-self.source[1],2))
            if new_length != 0:
                self.length = new_length
            else:
                logging.warning("Line: new calculated length is 0")
        
    def destination(self):
        """ Calculate the destination vector of the line """
        ex = self.source[0] + (self.length * self.direction[0])
        ey = self.source[1] + (self.length * self.direction[1])
        return np.array([ex,ey])

    def bounds(self):
        if self.swapped:
            return np.row_stack((self.destination(),self.source))
        else:
            return np.row_stack((self.source,self.destination()))
    
    @staticmethod
    def newLine(a,b,bbox):
        """ Create a new line from two vertices """
        #Calculate the line parameters:
        swapped = False
        d_a = utils.get_distance(np.array([[a.x,a.y]]),CENTRE)
        d_b = utils.get_distance(np.array([[b.x,b.y]]),CENTRE)
        aInBBox = a.within(bbox)
        bInBBox = b.within(bbox)
        if d_b < d_a and bInBBox:
            logging.debug("Swapping vertices for line creation, source is now: {}".format(b))
            temp = a
            a = b
            b = temp
            swapped = True
        vx = b.x - a.x
        vy = b.y - a.y
        l = sqrt(pow(vx,2) + pow(vy,2))
        if l != 0:
            scale = 1/l
        else:
            scale = 0
        dx = vx * scale
        dy = vy * scale
        cx = a.x + (dx * l)
        cy = a.y + (dy * l)
        return Line(a.x,a.y,dx,dy,l,swapped=swapped)
    
        
        
#--------------------
class HalfEdge:
    """ A Canonical Half-Edge. Has an origin point, and a twin half-edge for its end point,
        Auto-maintains counter-clockwise vertex order with it's twin
    """
    currentIndex = 0
    
    def __init__(self, origin, twin=None):
        self.origin = origin
        if origin:
            self.origin.registerHalfEdge(self)
        self.twin = twin
        self.face = None
        self.next = None
        self.prev = None
        self.index = HalfEdge.currentIndex
        HalfEdge.currentIndex += 1

        #Additional:
        self.markedForCleanup = False
        self.constrained = False
        self.drawn = False
        self.fixed = False
        
    def __str__(self):
        return "HalfEdge: {} - {}".format(self.origin,self.twin.origin)

    def __lt__(self,other):
        """
            Comparison of the origin and other.origin, from ciamej's stack overflow answer
            sorts clockwise relative to the centre of the face
            from: stackoverflow.com/questions/6989100
        """
        if not isinstance(other,HalfEdge):
            raise Exception("Trying to sort a Halfedge with something else")
        if self.origin is None or other.origin is None:
            raise Exception("Trying to compare against ill-formed edges")
        logging.debug("---- Half Edge Comparision")
        logging.debug("HELT: {} - {}".format(self.index,other.index))
        retValue = False
        #flip y axis for ease
        centre = [0,1] + (self.face.getCentre() * [1,-1])
        a = [0,1] + (self.origin.toArray() * [1,-1])
        b = [0,1] + (other.origin.toArray() * [1,-1])
        logging.debug("Comp: {}, {}, {}".format(centre,a,b))
        #offsets:
        o_a = a - centre
        o_b = b - centre
        #this, while in the SO answer, does not seem sufficient:
        # if o_a[0] >= 0 and o_b[0] < 0:
        #     retValue = True
        # elif o_a[0] < 0 and o_b[0] >= 0:
        #     retValue = False
        # elif np.allclose([o_a[0],o_b[0]],0):
        #     logging.debug("On same horizontal line")
        #     if o_a[1] >= 0 or o_b[1] >= 0:
        #         retValue = a[1] > b[1]
        #     else:
        #         retValue = b[1] > a[1]
        #As such, only use the full cross product calculation
        if True:
            det = np.cross(o_a,o_b)
            logging.debug("Det Value: {}".format(det))
            if det < 0:
                retValue = True
            elif det > 0:
                retValue = False
            else:
                logging.debug("Comparing by distance to face centre")
                d1 = utils.get_distance(o_a,centre)
                d2 = utils.get_distance(o_b,centre)
                logging.debug("D1: {} \n D2: {}".format(d1,d2))
                retValue = (d1 < d2)[0]

        logging.debug("CW: {}".format(retValue))
        #invert because of inverted y axis
        return retValue


    def intersects_edge(self,bbox):
        """ Return an integer 0-3 of the edge of a bbox the line intersects
        0 : Left Vertical Edge
        1 : Top Horizontal Edge
        2 : Right Vertical Edge
        3 : Bottom Horizontal Edge
            bbox is [min_x,min_y,max_x,max_y]
        """
        if self.origin is None or self.twin.origin is None:
            raise Exception("Invalid line boundary test ")
        bbox = bbox + np.array([EPSILON,EPSILON,-EPSILON,-EPSILON])
        s = self.origin.toArray()
        e = self.twin.origin.toArray()
        #logging.debug("Checking edge intersection:\n {}\n {}\n->{}\n----".format(s,e,bbox))
        if s[0] <= bbox[0] or e[0] <= bbox[0]:
            return 0
        elif s[1] <= bbox[1] or e[1] <= bbox[1]:
            return 1
        elif s[0] >= bbox[2] or e[0] >= bbox[2]:
            return 2
        elif s[1] >= bbox[3] or e[1] >= bbox[3]:
            return 3
        else:
            return None #no intersection
    
    def connections_align(self,other):
        if self.twin.origin is None or other.origin is None:
            raise Exception("Invalid connection test")
        if self.origin == other.origin \
           or self.twin.origin == other.twin.origin \
           or self.origin == other.twin.origin:
            logging.debug("Unexpected connection alignment")
            #praise Exception("Unexpected Connection Alignment")
        return self.twin.origin == other.origin
        
    def isConstrained(self):
        return self.constrained or self.twin.constrained
        
    def setConstrained(self):
        self.constrained = True
        self.twin.constrained = True
        
    def within(self,bbox):
        """ Check that both points in an edge are within the bbox """
        return self.origin.within(bbox) and self.twin.origin.within(bbox)

    def outside(self,bbox):
        return self.origin.outside(bbox) and self.twin.origin.outside(bbox)

    def constrain(self,bbox):
        """ Constrain the half-edge to be with a bounding box of [min_x,min_y,max_x,max_y]"""
        if self.origin is None:
            raise Exception("By this stage all edges should have an origin")
        if self.twin is None: 
            raise Exeption("Can't bound a single half-edge")
        if self.twin.origin is None:
            raise Exception("By this stage all edges should be finite")

        #Convert to an actual line representation, for intersection
        logging.info("Constraining {} - {}".format(self.index,self.twin.index))
        asLine = Line.newLine(self.origin,self.twin.origin,bbox)
        asLine.constrain(*bbox)
        return asLine.bounds()
        
        
    def addVertex(self,vertex):
        if self.origin is None:
            self.origin = vertex
            self.origin.registerHalfEdge(self)
        elif self.twin.origin is None:
            self.twin.origin = vertex
            self.twin.origin.registerHalfEdge(self)
        else:
            raise Exception("trying to add a vertex to a full edge")

    def fixup(self):
        """ Fix the clockwise/counter-clockwise property of the edge """
        #Swap to maintain counter-clockwise property
        if self.fixed or self.twin.fixed:
            logging.debug("Fixing an already fixed line")
            #return
        if self.origin is not None and self.twin.origin is not None:
            #angle = utils.angle_between_points(self.origin.toArray(),
            #                                   self.twin.origin.toArray())
            cmp = self < self.twin
            otherCmp = self.twin < self
            logging.debug("Cmp Pair: {} - {}".format(cmp,otherCmp))
            if cmp != otherCmp:
                logging.info("Mismatched Indices: {}-{}".format(self.index,self.twin.index))
                logging.info("Mismatched: {} - {}".format(self,self.twin))
                raise Exception("Mismatched orientations")
            logging.debug("CMP: {}".format(cmp))
            if cmp:
                logging.warning("Swapping the vertices of line {} and {}".format(self.index,self.twin.index))
                #unregister
                self.twin.origin.unregisterHalfEdge(self.twin)
                self.origin.unregisterHalfEdge(self)
                #cache
                temp = self.twin.origin
                #switch
                self.twin.origin = self.origin
                self.origin = temp
                #re-register
                self.twin.origin.registerHalfEdge(self.twin)
                self.origin.registerHalfEdge(self)

                reCheck = self < self.twin
                reCheck_opposite = self.twin < self
                if reCheck or reCheck_opposite:
                    logging.warn("Re-Orientation failed")
                    raise Exception("Re-Orientation failed")
                
            self.fixed = True
            self.twin.fixed = True

    def clearVertices(self):
        """ move vertices from the edge, clearing the vertex->edge references as well   """
        v1 = self.origin
        v2 = self.twin.origin
        self.origin = None
        self.twin.origin = None
        if v1:
            v1.unregisterHalfEdge(self)
        if v2:
            v2.unregisterHalfEdge(self.twin)
            
    def swapFaces(self):
        if not self.face and self.twin.face:
            raise Exception("Can't swap faces when at least one is missing")
        oldFace = self.face
        self.face = self.twin.face
        self.twin.face = oldFace
        
    def setNext(self,nextEdge):
        self.next = nextEdge
        self.next.prev = self
        
    def setPrev(self,prevEdge):
        self.prev = prevEdge
        self.prev.next = self
        
    def getVertices(self):
        return (self.origin,self.twin.origin)

    def isInfinite(self):
        return self.origin is None or self.twin is None or self.twin.origin is None
    
#--------------------
class Face(object):
    """ A Face with a start point for its outer component list, and all of its inner components """

    currentIndex = 0
    
    def __init__(self):
        #Starting point for bounding edges, going anti-clockwise
        self.outerComponent = None
        #Clockwise inner loops
        self.innerComponents = []
        self.edgeList = []
        self.index = Face.currentIndex
        Face.currentIndex += 1

    def removeEdge(self,edge):
        self.innerComponents.remove(edge)
        self.edgeList.remove(edge)
        
    def get_bbox(self):
        vertexPairs = [x.getVertices() for x in self.edgeList]
        vertexArrays = [(x.toArray(),y.toArray()) for x,y in vertexPairs if x is not None and y is not None]
        if len(vertexArrays) == 0:
            return np.array([[0,0],[0,0]])
        allVertices = np.array([x for (x,y) in vertexArrays for x in (x,y)])
        bbox = np.array([[allVertices[:,0].min(), allVertices[:,1].min()],
                         [allVertices[:,0].max(), allVertices[:,1].max()]])
        #logging.debug("Bbox source : {}".format(allVertices))
        #logging.debug("Bbox found  : {}".format(bbox))
        return bbox

    
    def getCentre(self):
        bbox = self.get_bbox()
        #max - min /2
        norm = bbox[1,:] + bbox[0,:]
        centre = norm * 0.5
        #logging.debug("Centre of {} : {}".format(self.index,centre))
        return centre
        
        
    def getEdges(self):
        return self.edgeList
        
    def add_edge(self,edge):
        self.innerComponents.append(edge)
        self.edgeList.append(edge)

    def sort_edges(self):
        """ Order the edges anti-clockwise, by starting point """
        logging.debug("Sorting edges")
        self.edgeList = sorted(self.edgeList)
        self.edgeList.reverse()
        logging.debug("Sorted edges: {}".format([str(x.index) for x in self.edgeList])) 
        
        
#--------------------
class DCEL(object):
    """ The total DCEL data structure, stores vertices, edges, and faces """
    
    def __init__(self,bbox=[-200,-200,200,200]):
        self.vertices  = []
        self.faces     = []
        self.halfEdges = []
        self.vertex_quad_tree = pyqtree.Index(bbox=bbox)
        self.bbox = bbox
        
    def __str__(self):
        verticesDescription = "Vertices: num: {}".format(len(self.vertices))
        edgesDescription = "HalfEdges: num: {}".format(len(self.halfEdges))
        facesDescription = "Faces: num: {}".format(len(self.faces))

        allVertices = [x.getVertices() for x in self.halfEdges]
        flattenedVertices = [x for (x,y) in allVertices for x in (x,y)]
        setOfVertices = set(flattenedVertices)
        vertexSet = "Vertex Set: num: {}/{}".format(len(setOfVertices),len(flattenedVertices))

        infiniteEdges = [x for x in self.halfEdges if x.isInfinite()]
        infiniteEdgeDescription = "Infinite Edges: num: {}".format(len(infiniteEdges))

        completeEdges = []
        for x in self.halfEdges:
            if not x in completeEdges and x.twin not in completeEdges:
                completeEdges.append(x)

        completeEdgeDescription = "Complete Edges: num: {}".format(len(completeEdges))

        edgelessVertices = [x for x in self.vertices if x.isEdgeless()]
        edgelessVerticesDescription = "Edgeless vertices: num: {}".format(len(edgelessVertices))

        edgeCountForFaces = [str(len(f.innerComponents)) for f in self.faces]
        edgeCountForFacesDescription = "Edge Counts for Faces: {}".format("-".join(edgeCountForFaces))
        
        return "\n".join(["---- DCEL Description: ",verticesDescription,edgesDescription,facesDescription,vertexSet,infiniteEdgeDescription,completeEdgeDescription,edgelessVerticesDescription,edgeCountForFacesDescription,"----\n"])


    
    def newVertex(self,x,y):
        """ Get a new vertex, or reuse an existing vertex """
        newVertex = Vertex(x,y)
        matchingVertices = self.vertex_quad_tree.intersect(newVertex.bbox())
        if matchingVertices:
            #a good enough vertex exists
            newVertex = matchingVertices.pop()
            logging.debug("Found a matching vertex: {}".format(newVertex))
        else:
            #no matching vertex, add this new one
            logging.debug("No matching vertex, storing: {}".format(newVertex))
            self.vertices.append(newVertex)
            self.vertex_quad_tree.insert(item=newVertex,bbox=newVertex.bbox())
        return newVertex
    
    def newEdge(self,originVertex,twinVertex,face=None,twinFace=None,prev=None,prev2=None):
        """ Get a new half edge pair, after specifying its start and end.
            Can set the faces, and previous edges of the new edge pair. 
            Returns the outer edge
        """
        e1 = HalfEdge(originVertex,None)
        e2 = HalfEdge(twinVertex,e1)
        e1.twin = e2 #fixup
        if face:
            e1.face = face
            face.add_edge(e1)
        if twinFace:
            e2.face = twinFace
            twinFace.add_edge(e2)
        if prev:
            e1.prev = prev
            prev.next = e1
        if prev2:
            e2.prev = prev2
            prev2.next = e2
        self.halfEdges.extend([e1,e2])
        return e1
        
    def newFace(self):
        """ Creates a new face to link edges """
        newFace = Face()
        self.faces.append(newFace)
        return newFace

    def linkEdgesTogether(self,edges):
        for i,e in enumerate(edges):
            e.prev = edges[i-1]
            e.prev.next = e
    
    def setFaceForEdgeLoop(self,face,edge,isInnerComponentList=False):
        """ For a face and a list of edges, link them together
            If the edges are the outer components, just put the first edge in the face,
            Otherwise places all the edges in the face """
        start = edge
        current = edge.next
        if isInnerComponentList:
            face.innerComponents.append(start)
        else:
            face.outerComponent = start
        start.face = face
        while current is not start and current.next is not None:
            current.face = face
            current = current.next
            if isInnerComponentList:
                face.innerComponents.append(current)

                
    def orderVertices(self,focus,vertices):
        """ Given a focus point and a list of vertices, sort them
            by the counter-clockwise angle position they take relative """
        relativePositions = [[x-focus[0],y-focus[1]] for x,y in vertices]        
        angled = [(atan2(yp,xp),x,y) for xp,yp,x,y in zip(relativePositions,vertices)]
        sortedAngled = sorted(angled)        
        return sortedAngled

    def constrain_half_edges(self,bbox):
        """ For each halfedge, shorten it to be within the bounding box  """
        logging.debug("\n---------- Constraint Checking")
        numEdges = len(self.halfEdges)
        for e in self.halfEdges:
            logging.debug("\n---- Checking constraint for: {}/{} {}".format(e.index,numEdges,e))
            #if both vertices are within the bounding box, don't touch
            if e.isConstrained():
                continue
            if e.within(bbox):
                continue
            #if both vertices are out of the bounding box, clean away entirely
            elif e.outside(bbox):
                e.markedForCleanup = True
                continue
            else:
                logging.debug("Constraining")
                #else constrain the point outside the bounding box:
                newBounds = e.constrain(bbox)
                orig_1,orig_2 = e.getVertices()
                e.clearVertices()
                v1 = self.newVertex(newBounds[0][0],newBounds[0][1])
                v2 = self.newVertex(newBounds[1][0],newBounds[1][1])
                #if not (v1.eq(orig_1) or v1.eq(orig_2) or v2.eq(orig_1) or v2.eq(orig_2)):
                if not (v1 == orig_1 or v1 == orig_2 or v2 == orig_1 or v2 == orig_2):
                    logging.debug("Vertex Changes upon constraint:")
                    logging.debug(" Originally: {}, {}".format(orig_1,orig_2))
                    logging.debug(" New: {}, {}".format(v1,v2))
                    raise Exception("One vertex shouldn't change")
                
                e.addVertex(v1)
                e.addVertex(v2)
                e.setConstrained()
                logging.debug("Result: {}".format(e)) 

        #remove edges marked for cleanup
        self.purge_edges()
        #remove vertices with no associated edges
        self.purge_vertices()

    def purge_infinite_edges(self):
        logging.debug("Purging infinite edges")
        edges_to_purge = [x for x in self.halfEdges if x.isInfinite()]
        for e in edges_to_purge:
            e.clearVertices()
            self.halfEdges.remove(e)
            e.face.removeEdge(e)
        
        
    def purge_edges(self):
        logging.debug("Purging edges marked for cleanup")
        edges_to_purge = [x for x in self.halfEdges if x.markedForCleanup]
        for e in edges_to_purge:
            e.clearVertices()
            self.halfEdges.remove(e)
            e.face.removeEdge(e)

        
    def purge_vertices(self):
        used_vertices = [x for x in self.vertices if len(x.halfEdges) > 0]
        self.vertices = used_vertices
        

    def complete_faces(self,bbox):
        """ Verify each face, connecting non-connected edges, taking into account
            corners of the bounding box passed in, which connects using a pair of edges        
        """
        logging.info("---------- Completing faces")
        if bbox is None:
            raise Exception("Completing faces requires a bbox provided")
        for f in self.faces:
            logging.debug("Completing face: {}".format(f.index))
            #sort edges going anti-clockwise
            f.sort_edges()
            edgeList = f.getEdges().copy()
            #reverse to allow popping off
            edgeList.reverse()
            first_edge = edgeList[-1]
            while len(edgeList) > 1:
                #pop off in anti-clockwise order
                current_edge = edgeList.pop()
                nextEdge = edgeList[-1]
                logging.debug("---- Edge Pair: {} - {}".format(current_edge.index,nextEdge.index))
                if not current_edge.connections_align(nextEdge):
                    logging.debug("Edges do not align:\n\t e1: {} \n\t e2: {}".format(current_edge.twin.origin,nextEdge.origin))
                    #if they intersect with different bounding walls, they need a corner
                    intersect_1 = current_edge.intersects_edge(bbox)
                    intersect_2 = nextEdge.intersects_edge(bbox)
                    logging.debug("Intersect Values: {} {}".format(intersect_1,intersect_2))
                    if intersect_1 is None or intersect_2 is None:
                        logging.debug("Non- side intersecting lines")
                    if intersect_1 == intersect_2 or intersect_1 is None or intersect_2 is None:
                        logging.debug("Intersects match, creating a simple edge between: {}={}".format(current_edge.index,nextEdge.index))
                        #connect together with simple edge
                        newEdge = self.newEdge(current_edge.twin.origin,nextEdge.origin,face=f,prev=current_edge)
                        newEdge.next = nextEdge
                        nextEdge.prev = newEdge
                    else:
                        logging.debug("Creating a corner edge connection between: {}={}".format(current_edge.index,nextEdge.index))
                        #connect via a corner
                        newVertex = self.create_corner_vertex(intersect_1,intersect_2,bbox)
                        logging.debug("Corner Edge: {}".format(newVertex))
                        newEdge_1 = self.newEdge(current_edge.twin.origin,newVertex,face=f,prev=current_edge)
                        newEdge_2 = self.newEdge(newVertex,nextEdge.origin,face=f,prev=newEdge_1)
                        newEdge_1.next = newEdge_2
                        newEdge_2.prev = newEdge_1
                        newEdge_2.next = nextEdge
                        nextEdge.prev = newEdge_2
                
            #at this point, only the last hasn't been processed
            #as above, but:
            logging.debug("Checking final edge pair")
            current_edge = edgeList.pop()
            if not current_edge.connections_align(first_edge):
                intersect_1 = current_edge.intersects_edge(bbox)
                intersect_2 = first_edge.intersects_edge(bbox)
                if intersect_1 is None or intersect_2 is None:
                    logging.debug("Edge Intersection is None")
                elif intersect_1 == intersect_2:
                    logging.debug("Intersects match, creating final simple edge")
                    #connect with simple edge
                    newEdge = self.newEdge(current_edge.twin.origin,first_edge.origin,face=f,prev=current_edge)
                    newEdge.next = nextEdge
                    first_edge.prev = newEdge
                else:
                    logging.debug("Creating final corner edge connection between: {}={}".format(current_edge.index,first_edge.index))
                    #connect via a corner
                    newVertex = self.create_corner_vertex(intersect_1,intersect_2,bbox)
                    newEdge_1 = self.newEdge(current_edge.twin.origin,newVertex,face=current_edge.face,prev=current_edge)
                    newEdge_2 = self.newEdge(newVertex,first_edge.origin,face=current_edge.face,prev=newEdge_1)
                    #newEdge_1 = self.newEdge(current_edge.twin.origin,newVertex,face=current_edge.face,prev=current_edge)
                    #newEdge_2 = self.newEdge(newVertex,nextEdge.origin,face=current_edge.face,prev=newEdge_1)
                    newEdge_1.next = newEdge_2
                    newEdge_2.prev = newEdge_1
                    newEdge_2.next = first_edge
                    first_edge.prev = newEdge_2

            logging.debug("Final sort of face: {}".format(f.index))
            f.sort_edges()
            logging.debug("Result: {}".format([x.index for x in f.getEdges()]))
            logging.debug("----")

    def create_corner_vertex(self,e1,e2,bbox):
        """ Given two edges, create the vertex that corners them """
        if e1 == e2:
            raise Exception("Corner Edge Creation Error: edges are equal")
        if e1 % 2 == 0: #create the x vector
            v1 = np.array([bbox[e1],0])
        else:
            v1 = np.array([0,bbox[e1]])
        if e2 % 2 == 0: #create the y vector
            v2 = np.array([bbox[e2],0])
        else:
            v2 = np.array([0,bbox[e2]])
        #add together to get corner
        v3 = v1 + v2
        return self.newVertex(*v3)

    def fixup_halfedges(self):
        logging.debug("---- Fixing order of vertices for each edge")
        for e in self.halfEdges:
            e.fixup()

    def verify_edges(self):
        #make sure every halfedge is only used once
        logging.info("Verifying edges")
        troublesomeEdges = [] #for debugging
        usedEdges = {}
        for f in self.faces:
            for e in f.edgeList:
                if e.isInfinite():
                    raise Exception("Edge {} is infinite when it shouldn't be".format(e.index))
                if e < e.twin:
                    raise Exception("Edge {} is not anti-clockwise".format(e.index))
                    #raise Exception("Edge {} is not anti clockwise".format(e.index))
                if e.index not in usedEdges:
                    usedEdges[e.index] = f.index
                else:
                    raise Exception("Edge {} in {} already used in {}".format(e.index,f.index,usedEdges[e.index]))
        logging.info("Edges verified")
        return troublesomeEdges
        
