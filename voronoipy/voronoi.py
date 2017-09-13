import numpy as np
import numpy.random as random
import pyqtree
import IPython
import heapq
import pickle
import logging as root_logger
import sys
from os.path import isfile
from string import ascii_uppercase
from numpy.linalg import det
from math import pi, sin, cos, nan

import cairo_utils as utils
from cairo_utils import Parabola
from cairo_utils.beachline import BeachLine, NilNode, Node
from cairo_utils.beachline.operations import Directions
from cairo_utils.dcel import DCEL

from .Events import SiteEvent, CircleEvent

logging = root_logger.getLogger(__name__)
#If true, will draw each frame as infinite lines are made finite
DEBUG_INFINITE_RESOLUTION = False

#SAVE FILE:
SAVENAME = "graph_data.pkl"

#COLOURS and RADI:
COLOUR = [0.2,0.1,0.6,1.0]
COLOUR_TWO = [1.0,0.2,0.4,0.5]
SITE_COLOUR = [1,0,0,1]
SITE_RADIUS = 0.002
CIRCLE_COLOUR = [1,1,0,1]
CIRCLE_COLOUR_INACTIVE = [0,0,1,1]
CIRCLE_RADIUS = 0.005
BEACH_LINE_COLOUR = [0,1,0]
BEACH_LINE_COLOUR2 = [1,1,0]
BEACH_NO_INTERSECT_COLOUR = [1,0,0,1]
BEACH_RADIUS = 0.002
SWEEP_LINE_COLOUR = [0,0,1,1]
LINE_WIDTH = 0.002
#:
BBOX = np.array([0,0,1,1]) #the bbox of the final image
EPSILON = sys.float_info.epsilon

class Voronoi:
    """ Creates a random selection of points, and step by step constructs
        a voronoi diagram
    """
    def __init__(self,sizeTuple,num_of_nodes=10, bbox=BBOX, save_name=SAVENAME):
        self.current_step = 0
        self.sX = sizeTuple[0]
        self.sY = sizeTuple[1]
        self.nodeSize = num_of_nodes
        #Heap of site/circle events
        self.events = []
        #backup of the original sites
        self.sites = []
        #backup of all circle events
        self.circles = []
        #storage of breakpoint tuples -> halfedge
        self.halfEdges = {}
        #The bbox of the diagram
        self.bbox = bbox

        #The Beach Line Data Structure
        self.beachline = None
        #The sweep line position
        self.sweep_position = None
        #The output voronoi diagram as a DCEL
        self.dcel = DCEL(bbox=BBOX)

        self.save_file_name = save_name
        
    def reset(self):
        """ Reset the internal data structures """
        self.dcel = DCEL(bbox=BBOX)
        self.events = []
        self.circles = []
        self.halfEdges = []
        self.beachline =
        self.sweep_position = None
        self.beachline = BeachLine()
        self.current_step = 0

        
    def initGraph(self,data=None,rerun=False):
        """ Create a graph of initial random sites """
        logging.debug("Initialising graph")
        self.reset()
        
        values = data
        if values is None and not rerun:
            values = self.load_graph()
        #create a (n,2) array of coordinates for the sites, if no data has been loaded
        if values is None or len(values) != self.nodeSize:
            logging.debug("Creating values")
            for n in range(self.nodeSize):
                newSite = random.random(2)
                if values is None:
                    values = np.array([newSite])
                else:
                    values = np.row_stack((values,newSite))

        #create the site events:
        usedCoords = []
        for site in values:
            if (site[0],site[1]) in usedCoords:
                logging.warn("Skipping: {}".format(site))
                continue
            futureFace = self.dcel.newFace(site[0],site[1])
            event = SiteEvent(site,face=futureFace)
            heapq.heappush(self.events,event)
            self.sites.append(event)
            usedCoords.append((site[0],site[1]))

        #Save the nodes
        if not rerun:
            self.save_graph(values)
        return values

    def relax(self):
        """ Having calculated the voronoi diagram, use the centroids of 
            the faces instead of the sites, and retrun the calculation
        """
        assert(not bool(self.events))
        newSites = [x.getAvgCentroid() for x in self.dcel.faces]
        logging.info("Num of Faces: {}".format(len(newSites)))
        self.initGraph(data=newSites,rerun=True)
        self.calculate_to_completion()
        
    def calculate_to_completion(self):
        """ Calculate the entire voronoi for all points """
        finished = False
        while not finished:
            logging.debug("Calculating step: {}".format(i))
            finished = self._calculate(i)
            self.current_step += 1

    def _calculate(self,i):
        """ Calculate the next step of the voronoi diagram,
            Return True on completion, False otherwise
        """
        if not bool(self.events): #finished calculating
            return True
        ##handle site / circle event
        event = heapq.heappop(self.events)
        #update the sweep position
        self.sweep_position = event
        logging.debug("Sweep position: {}".format(self.sweep_position.loc))
        #update the arcs:
        self._update_arcs(self.sweep_position.y())
        #handle the event:
        if isinstance(event,SiteEvent):
            self._handleSiteEvent(event)
        elif isinstance(event,CircleEvent):
            if event.active:
                self._handleCircleEvent(event)
            else:
                logging.debug("-------------------- Skipping deactivated circle event")
                logging.debug(event)
        else:
            raise Exception("Unrecognised Event")
        return False 
        
    def finalise_DCEL(self):
        """ Cleanup the DCEL of the voronoi diagram, 
            completing faces and constraining to a bbox 
        """
        assert(not bool(self.events))
        logging.debug("---------- Finalising DCEL")
        logging.debug(self.dcel)
        #Not a pure DCEL operation as it requires curve intersection:
        self._complete_edges()
        #purge obsolete DCEL data:
        self.dcel.purge_infinite_edges()
        self.dcel.constrain_half_edges(bbox=BBOX)
        self.dcel.fixup_halfedges()
        #todo: connect edges of faces together
        self._complete_faces()
        return self.dcel

    #FORTUNE METHODS
    def _handleSiteEvent(self,event):
        """
        provided with a site event, add it to the beachline in the appropriate place
        then update/remove any circle events that trios of arcs generate
        """
        assert(isinstance(event, SiteEvent))
        logging.debug("Handling Site Event: {}".format(event))
        #for visualisation: add an arc
        new_arc = Parabola(*event.loc,self.sweep_position.y())
        #if beachline is empty: add and return
        if self.beachline.isEmpty():
            newNode = self.beachline.insert(new_arc)
            newNode.data['face'] = event.face
            return
                
        #get the x position of the event
        xPos = new_arc.fx
        #search for the breakpoint interval of the beachline
        closest_arc_node,dir = self.beachline.search(xPos)
        
        logging.debug("Closest Arc Triple: {} *{}* {}".format(closest_arc_node.get_predecessor(),
                                                             closest_arc_node,
                                                             closest_arc_node.get_successor()))
        logging.debug("Direction: {}".format(dir))

        #remove false alarm circle events
        if closest_arc_node.left_circle_event is not None:
            self._delete_circle_event(closest_arc_node.left_circle_event)
        if closest_arc_node.right_circle_event is not None:
            self._delete_circle_event(closest_arc_node.right_circle_event)
            
        #split the beachline
        if dir is Directions.CENTRE or dir is Directions.RIGHT:
            new_node = self.beachline.insert_successor(closest_arc_node,new_arc)
            duplicate_node = self.beachline.insert_successor(new_node,closest_arc_node.value)
        else:
            new_node = self.beachline.insert_predecessor(closest_arc_node,new_arc)
            duplicate_node = self.beachline.insert_predecessor(new_node,closest_arc_node.value)

        assert(isinstance(new_node, Node))
            
        newTriple = [closest_arc_node.value.id,new_node.value.id,duplicate_node.value.id]
        tripleString = "-".join([ascii_uppercase[x] if x < 26 else str(x) for x in newTriple])
        logging.debug("Split {} into {}".format(str(newTriple[0]),tripleString))

        #add in the face as a data point for the new node and duplicate
        new_node.data['face'] = event.face
        duplicate_node.data['face'] = closest_arc_node.data['face']
        
        #create a half-edge pair between the two nodes, store the tuple (edge,arc)
        face_a = new_node.data['face']
        if 'face' in closest_arc_node.data:
            face_b = closest_arc_node.data['face']
        else:
            raise Exception("A Face can't be found for a node")
        
        #set the origin points to be undefined
        #link the edges with the faces associated with the sites
        logging.debug("Adding edge")
        newEdge = self.dcel.newEdge(None,None,face=face_b,twinFace=face_a)
        #newEdge = self.dcel.newEdge(None,None,face=face_a,twinFace=face_b)

        #if there was an edge of closest_arc -> closest_arc.successor: update it
        #because closest_arc is not adjacent to successor any more, duplicate_node is
        dup_node_succ = duplicate_node.get_successor()
        if dup_node_succ:
            e1 = self._getEdge(closest_arc_node,dup_node_succ)
            if e1:
                self._removeEdge(closest_arc_node,dup_node_succ)
                self._storeEdge(e1,duplicate_node,dup_node_succ)
                
        
        logging.debug("Linking edge from {} to {}".format(closest_arc_node,new_node))
        self._storeEdge(newEdge,closest_arc_node,new_node)
        logging.debug("Linking r-edge from {} to {}".format(new_node,duplicate_node))
        self._storeEdge(newEdge.twin,new_node,duplicate_node)
        
        #create circle events:
        self._calculate_circle_events(new_node)
        
    def _handleCircleEvent(self,event):
        """
        provided a circle event, add a new vertex to the dcel, 
        then update the beachline to connect the two sides of the arc that has disappeared
        """
        assert(isinstance(event, CircleEvent))
        logging.debug("Handling Circle Event: {}".format(event))
        #remove disappearing arc from tree
        #and update breakpoints, remove false alarm circle events
        node = event.source
        pre = node.get_predecessor()
        suc = node.get_successor()

        if node.left_circle_event:
            self._delete_circle_event(node.left_circle_event)
        if node.right_circle_event:
            self._delete_circle_event(node.right_circle_event)
        logging.debug("attempting to remove pre-right circle events for: {}".format(pre))
        if pre != NilNode and pre.right_circle_event is not None:
            self._delete_circle_event(pre.right_circle_event)
        logging.debug("Attempting to remove succ-left circle events for: {}".format(suc))
        if suc != NilNode and suc.left_circle_event is not None:
            self._delete_circle_event(suc.left_circle_event)
        
        #add the centre of the circle causing the event as a vertex record
        logging.debug("Creating Vertex")
        newVertex = self.dcel.newVertex(event.vertex[0],event.vertex[1])

        #attach the vertex as a defined point in the half edges for the three faces,
        #these faces are pre<->node and node<->succ

        e1 = self._getEdge(pre,node)
        e2 = self._getEdge(node,suc)
        if e1:
            logging.debug("Adding vertex to {}-{}".format(pre,node))
            e1.addVertex(newVertex)
        else:
            logging.debug("No r-edge found for {}-{}".format(pre,node))
            
        if e2:
            logging.debug("Adding vertex to {}-{}".format(node,suc))
            e2.addVertex(newVertex)
        else:
            logging.debug("No r-edge found for {}-{}".format(node,suc))
            
        if not 'face' in pre.data or not 'face' in suc.data:
            raise Exception("Circle event on faceless nodes")
            
        #create two half-edge records for the new breakpoint of the beachline
        logging.debug("Creating a new half-edge {}-{}".format(pre,suc))
        newEdge = self.dcel.newEdge(None,newVertex,face=suc.data['face'],twinFace=pre.data['face'])
        #newEdge = self.dcel.newEdge(None,newVertex,face=pre.data['face'],twinFace=suc.data['face'])

        if e1:
            e1.setPrev(newEdge)
        if e2:
            e2.setNext(newEdge)
        
        #store the new edge, but only for the open breakpoint
        #the breakpoint being the predecessor and successor, now partners following
        #removal of the middle node above in this function  
        self._storeEdge(newEdge,pre,suc)

        #delete the node, no longer needed as the arc has reduced to 0
        self.beachline.delete_node(node)

        #recheck for new circle events
        if pre:
            self._calculate_circle_events(pre,left=False)
        if suc:
            self._calculate_circle_events(suc,right=False)
        

    ## UTILITY METHODS ----------------------------------------
    #-------------------- Import / Export
    def save_graph(self,values):
        with open(self.save_file_name,'wb') as f:
            pickle.dump(values,f)
        
    def load_graph(self):
        if isfile(self.save_file_name):
            with open(self.save_file_name,'rb') as f:
                return pickle.load(f)
        
    #-------------------- Fortune Methods
    def _calculate_circle_events(self,node,left=True,right=True):
        """
        Given an arc node, get the arcs either side, and determine if/when it will disappear
        """
        logging.debug("Calculating circle events for: {}".format(node))
        #Generate a circle event for left side, and right side
        left_triple = self.beachline.get_predecessor_triple(node)
        right_triple = self.beachline.get_successor_triple(node)
        #Calculate chords and determine circle event point:
        #add circle event to events and the relevant leaf
        if left_triple:
            logging.debug("Calc Left Triple: {}".format("-".join([str(x) for x in left_triple])))
        if left and left_triple and left_triple[0].value != left_triple[2].value:
            left_points = [x.value.get_focus() for x in left_triple]
            left_circle = utils.get_circle_3p(*left_points)
            if left_circle and not utils.isClockwise(*left_points,cartesian=True):
                left_circle_loc = utils.get_lowest_point_on_circle(*left_circle)
                #check the l_t_p/s arent in this circle
                #note: swapped this to add on the right ftm
                self._add_circle_event(left_circle_loc,left_triple[1],left_circle[0],left=False)
            else:
                logging.debug("Left circle response: {}".format(left_circle))

        if right_triple:
            logging.debug("Calc Right Triple: {}".format("-".join([str(x) for x in right_triple])))
        if right and right_triple and right_triple[0].value != right_triple[2].value:
            right_points = [x.value.get_focus() for x in right_triple]
            right_circle = utils.get_circle_3p(*right_points)
            if right_circle and not utils.isClockwise(*right_points,cartesian=True):
                right_circle_loc = utils.get_lowest_point_on_circle(*right_circle)
                #note: swapped this to add on the left ftm
                self._add_circle_event(right_circle_loc,right_triple[1],right_circle[0])
            else:
                logging.debug("Right circle response: {}".format(right_circle))

    def _update_arcs(self,d):
        """ Trigger the update of all stored arcs with a new frontier line position """
        self.beachline.update_arcs(d)

    #-------------------- DCEL Completion
    def _complete_edges(self):
        """ get any infinite edges, and complete their intersections """
        logging.debug("\n---------- Infinite Edges Completion")
        i = 0
        #not is infinite, only actually caring about edges without a start
        i_pairs = [x for x in self.halfEdges.items() if x[1].origin is None]
        logging.debug("Infinite edges num: {}".format(len(i_pairs)))
        #----
        #i_pairs = [((breakpoint nodes),edge)]
        for ((a,b),c) in i_pairs:
            i += 1
            #a and b are nodes
            #logging.debug("All Infinite Check: {}".format(",".join([str(z.isInfinite()) for ((x,y),z) in i_pairs])))
            logging.debug("{} Infinite Edge resolution: {}-{}, infinite? {}".format(i,a,b,c.isInfinite()))
            if not c.isInfinite():
                logging.debug("An edge that is not infinite")
            intersection = a.value.intersect(b.value)
            if intersection is not None and intersection.shape[0] > 0:
                newVertex = self.dcel.newVertex(intersection[0,0],intersection[0,1])
                c.addVertex(newVertex)
                isInfiniteAfterIntersection = c.isInfinite()
                if isInfiniteAfterIntersection:
                    raise Exception("After modification is infinite")

                if DEBUG_INFINITE_RESOLUTION and surface and filename:
                    saveName = "{}_edge_completion_{}".format(filename,i)
                    utils.drawDCEL(self.ctx,self.dcel)
                    utils.write_to_png(surface,saveName)
            else:
                raise Exception("No intersections detected when completing an infinite edge")
            logging.debug('----')

    def _complete_faces(self,dcel=None):
        if dcel is None:
            dcel = self.dcel
        dcel.complete_faces(BBOX)
        return dcel

            
    #-------------------- Beachline Edge Interaction
    def _storeEdge(self,edge,bp1,bp2):
        """ Store an incomplete edge by the 2 pairs of nodes that define the breakpoints """
        if (bp1,bp2) in self.halfEdges.keys():
            raise Exception("Duplicating edge breakpoint")
        if edge in self.halfEdges.values():
            raise Exception("Duplicating edge store")
        self.halfEdges[(bp1,bp2)] = edge
        
    def _hasEdge(self,bp1,bp2):
        return (bp1,bp2) in self.halfEdges

    def _getEdge(self,bp1,bp2):
        if self._hasEdge(bp1,bp2):
            return self.halfEdges[(bp1,bp2)]
        else:
            return None

    def _removeEdge(self,bp1,bp2):
        if not self._hasEdge(bp1,bp2):
            raise Exception("trying to remove a non-existing edge")
        del self.halfEdges[(bp1,bp2)]

    #-------------------- Circle Event Interaction
    def _add_circle_event(self,loc,sourceNode,voronoiVertex,left=True):
        if loc[1] < self.sweep_position.y():# or np.allclose(loc[1],self.sweep_position.y()):
            logging.debug("Breaking out of add circle event: at/beyond sweep position")
            return
        #if True: #loc[1] > self.sweep_position[0]:
        if left:   
            event = CircleEvent(loc,sourceNode,voronoiVertex,i=self.current_step)
        else:
            event = CircleEvent(loc,sourceNode,voronoiVertex,left=False,i=self.current_step)
        logging.debug("Adding: {}".format(event))
        heapq.heappush(self.events,event)
        self.circles.append(event)

    def _delete_circle_event(self,event):
        logging.debug("Deleting Circle Event: {}".format(event))
        #self.events = [e for e in self.events if not e.nodeIs(event.source)]
        #heapq.heapify(self.events)
        event.deactivate()
        
            
