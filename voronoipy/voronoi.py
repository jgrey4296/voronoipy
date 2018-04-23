""" Voronoi.py : Contains the Voronoi Class, which calculates a graphics independent DCEL
    of a Voronoi Diagram.
"""
import numpy as np
import numpy.random as random
import heapq
import pickle
import logging as root_logger
import sys
import IPython
from os.path import isfile
from string import ascii_uppercase
from math import pi, sin, cos

import cairo_utils as utils
from cairo_utils import Parabola
from cairo_utils.beachline import BeachLine, NilNode, Node
from cairo_utils.beachline.operations import Directions
from cairo_utils.dcel import DCEL, HalfEdge, Face
from cairo_utils.math import get_distance_raw, bound_line_in_bbox, isClockwise

from .Events import SiteEvent, CircleEvent, VEvent
from .voronoi_drawing import Voronoi_Debug

logging = root_logger.getLogger(__name__)

#Constants and defaults
image_dir = "imgs"
SAVENAME = "graph_data.pkl"
BBOX = np.array([0,0,1,1]) #the bbox of the final image
EPSILON = sys.float_info.epsilon
MAX_STEPS = 100000
CARTESIAN = True

class Voronoi:
    """ Creates a random selection of points, and step by step constructs
        a voronoi diagram
    """
    def __init__(self, sizeTuple, num_of_nodes=10, bbox=BBOX, save_name=SAVENAME,
                 debug_draw=False, n=10, max_steps=MAX_STEPS):
        assert(isinstance(sizeTuple, tuple))
        assert(isinstance(bbox, np.ndarray))
        assert(bbox.shape == (4,))
        self.current_step = 0
        self.sX = sizeTuple[0]
        self.sY = sizeTuple[1]
        self.nodeSize = num_of_nodes
        self.max_steps = max_steps
        #Min Heap of site/circle events
        self.events = []
        #backup of the original sites
        self.sites = []
        #backup of all circle events
        self.circles = []
        #storage of breakpoint tuples -> halfedge
        self.halfEdges = {}
        #The bbox of the diagram
        self.bbox = bbox
        VEvent.offset = self.bbox[3] - self.bbox[1]
        
        #The Beach Line Data Structure
        self.beachline = None
        #The sweep line position
        self.sweep_position = None
        #The output voronoi diagram as a DCEL
        self.dcel = DCEL(bbox=bbox)

        #File name to pickle data to:
        self.save_file_name = save_name
        
        self.debug_draw = debug_draw
        self.debug = Voronoi_Debug(n, image_dir, self)
        
    #--------------------
    # PUBLIC METHODS
    #--------------------
    def reset(self):
        """ Reset the internal data structures """
        self.dcel = DCEL(bbox=self.bbox)
        self.events = []
        self.circles = []
        self.halfEdges = {}
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

        assert(values is None or isinstance(values, np.ndarray))
        
            
        #create a (n,2) array of coordinates for the sites, if no data has been loaded
        if values is None or len(values) != self.nodeSize:
            logging.debug("Generating values")
            for n in range(self.nodeSize):
                rndAmnt = random.random((1,2))
                #scale the new site
                scaler = self.bbox.reshape((2,2)).transpose()
                newSite = scaler[:,0] + (rndAmnt * (scaler[:,1] - scaler[:,0]))
                if values is None:
                    values = newSite
                else:
                    values = np.row_stack((values,newSite))

        #setup the initial site events:
        usedCoords = []
        for site in values:
            #Avoid duplications:
            if (site[0],site[1]) in usedCoords:
                logging.warn("Skipping Duplicate: {}".format(site))
                continue
            #Create an empty face for the site
            futureFace = self.dcel.newFace(site)
            event = SiteEvent(site,face=futureFace)
            heapq.heappush(self.events,event)
            self.sites.append(event)
            usedCoords.append((site[0],site[1]))

        #Save the nodes
        if not rerun:
            self.save_graph(values)
        return values

    def relax(self, amnt=0.5, faces=None):
        """ Having calculated the voronoi diagram, use the centroids of 
            the faces instead of the sites, and rerun the calculation.
        Can be passed in a subset of faces
        """
        assert(not bool(self.events))

        if faces is None:
            faces = self.dcel.faces
        #Figure out any faces that are excluded
        faceIndices = set([x.index for x in faces])
        otherFaceSites = np.array([x.site for x in self.dcel.faces if x.index not in faceIndices])
        #Get a line of origin - centroid
        lines = np.array([np.concatenate((x.site, x.getAvgCentroid())) for x in faces])
        #Move along that line toward the centroid
        newSites = np.array([utils.math.sampleAlongLine(*x, amnt)[0] for x in lines])
        #Combine with excluded faces
        if len(otherFaceSites) > 0 and len(newSites) > 0:
            totalSites = np.row_stack((newSites, otherFaceSites))
        elif len(newSites) > 0:
            totalSites = newSites
        else:
            totalSites = otherFaceSites
        assert(len(self.dcel.faces) == len(totalSites))
        #Setup the datastructures with the new sites
        self.initGraph(data=newSites,rerun=True)
        self.calculate_to_completion()
        
    def calculate_to_completion(self):
        """ Calculate the entire voronoi for all points """
        finished = False
        #Max Steps for a guaranteed exit
        while not finished and self.current_step < self.max_steps:
            logging.debug("Calculating step: {}".format(self.current_step))
            finished = self._calculate()
            if self.debug_draw:
                self.debug.draw_intermediate_states(self.current_step)
            self.current_step += 1

    def finalise_DCEL(self):
        """ Cleanup the DCEL of the voronoi diagram, 
            completing faces and constraining to a bbox 
        """
        if bool(self.events):
            logging.warning("Finalising with events still to process")
        logging.debug("-------------------- Finalising DCEL")
        logging.debug(self.dcel)
        #Not a pure DCEL operation as it requires curve intersection:
        self._complete_edges()
        self.dcel.purge()
        #modify or mark edges outside bbox
        tempbbox = self.bbox + np.array([100,100,-100,-100])
        self.dcel.constrain_to_bbox(tempbbox, force=True)
        self.dcel.purge()
        logging.debug("---------- Constrained to bbox")
        #ensure CCW ordering
        for f in self.dcel.faces:
            f.fixup(tempbbox)
        #cleanup faces
        logging.debug("---------- Fixed up faces")
        self.dcel.purge()
        logging.debug("---------- Purged 3")
        logging.debug(self.dcel)
        self.dcel.verify_all()
        return self.dcel

    def save_graph(self,values):
        with open(self.save_file_name,'wb') as f:
            pickle.dump(values,f)
        
    def load_graph(self):
        if isfile(self.save_file_name):
            with open(self.save_file_name,'rb') as f:
                return pickle.load(f)



    #--------------------
    # PRIVATE METHODS
    #--------------------
    def _calculate(self):
        """ Calculate the next step of the voronoi diagram,
            Return True on completion, False otherwise
        """
        if not bool(self.events): #finished calculating, early exit
            return True
        ##Get the next event
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
        
    
    #----------
    # MAIN VORONOI CALLS
    #----------
    def _handleSiteEvent(self,event):
        """
        provided with a site event, add it to the beachline in the appropriate place
        then update/remove any circle events that trios of arcs generate
        """
        assert(isinstance(event, SiteEvent))
        logging.debug("Handling Site Event: {}".format(event))
        #The new parabola made from the site
        new_arc = Parabola(*event.loc,self.sweep_position.y())
        #get the x position of the event
        xPos = new_arc.fx

        #if beachline is empty: add and return
        if self.beachline.isEmpty():
            newNode = self.beachline.insert(new_arc)
            newNode.data['face'] = event.face
            return

        #Otherwise, slot the arc between existing nodes
        closest_node, direction = self._get_closest_arc_node(xPos)
        self._remove_obsolete_circle_events(closest_node)
        new_node, duplicate_node  = self._split_beachline(direction,
                                                         closest_node,
                                                         new_arc,
                                                         event.face)
        
        #Create an edge between the two nodes, without origin points yet
        logging.debug("Adding edge on side: {}".format(direction))
        node_face = closest_node.data['face']
        if direction is Directions.LEFT:
            theFace = event.face
            twinFace = node_face
            nodePair = (new_node, closest_node)
        else:
            theFace = node_face
            twinFace = event.face
            nodePair = (closest_node, new_node)
        
        newEdge = self.dcel.newEdge(None, None, face=theFace, twinFace=twinFace)
        self._storeEdge(newEdge, *nodePair)
        self._cleanup_edges(direction, newEdge, new_node, closest_node, duplicate_node)

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
        assert('face' in pre.data)
        assert('face' in suc.data)

        self._delete_circle_events(node, pre, suc, event)
        
        #add the centre of the circle causing the event as a vertex record
        logging.debug("Creating Vertex")
        newVertex = self.dcel.newVertex(event.vertex)

        #attach the vertex as a defined point in the half edges for the three faces,
        #these faces are pre<->node and node<->succ

        e1 = self._getEdge(pre,node)
        e2 = self._getEdge(node,suc)

        #create two half-edge records for the new breakpoint of the beachline
        logging.debug("Creating a new half-edge {}-{}".format(pre,suc))
        newEdge = self.dcel.newEdge(newVertex, None,face=pre.data['face'],twinFace=suc.data['face'])

        if e1:
            #predecessor face
            logging.debug("Adding vertex to {}-{}".format(pre,node))
            assert(e1.face == pre.data['face'])
            assert(e1.twin.face == node.data['face'])
            e1.addVertex(newVertex)
            e1.addPrev(newEdge, force=True)
        else:
            logging.debug("No r-edge found for {}-{}".format(pre,node))
            IPython.embed(simple_prompt=True)
            
        if e2:
            #successor face
            logging.debug("Adding vertex to {}-{}".format(node,suc))
            assert(e2.twin.face == suc.data['face'])
            assert(e2.face == node.data['face'])
            e2.addVertex(newVertex)
            e2.twin.addNext(newEdge.twin, force=True)
        else:
            logging.debug("No r-edge found for {}-{}".format(node,suc))
            IPython.embed(simple_prompt=True)
        
        #store the new edge, but only for the open breakpoint
        #the breakpoint being the predecessor and successor, now partners following
        #removal of the middle node above in this function  
        self._storeEdge(newEdge,pre,suc)

        #delete the node, no longer needed as the arc has reduced to 0
        self.beachline.delete_node(node)
        
        #recheck for new circle events
        if pre:
            self._calculate_circle_events(pre,left=False, right=True)
        if suc:
            self._calculate_circle_events(suc,left=True, right=False)
        

    #----------
    # UTILITIES
    #----------        
    def _cleanup_edges(self, edge, new_node, node, duplicate_node):
        #if there was an edge of closest_arc -> closest_arc.successor: update it
        #because closest_arc is not adjacent to successor any more, duplicate_node is
        dup_node_succ = duplicate_node.get_successor()
        if dup_node_succ:
            e1 = self._getEdge(node,dup_node_succ)
            if e1:
                self._removeEdge(node,dup_node_succ)
                self._storeEdge(e1,duplicate_node,dup_node_succ)
                
        logging.debug("Linking edge from {} to {}".format(node,new_node))
        self._storeEdge(edge, node, new_node)
        logging.debug("Linking r-edge from {} to {}".format(new_node,duplicate_node))
        self._storeEdge(edge.twin, new_node, duplicate_node)
        
    def _get_closest_arc_node(self, xPos):
        #search for the breakpoint interval of the beachline
        closest_arc_node, direction = self.beachline.search(xPos)
        logging.debug("Closest Arc Triple: {} *{}* {}".format(closest_arc_node.get_predecessor(),
                                                             closest_arc_node,
                                                             closest_arc_node.get_successor()))
        logging.debug("Direction: {}".format(direction))
        return (closest_arc_node, direction)

    def _remove_obsolete_circle_events(self, node):
        #remove false alarm circle events
        if node.left_circle_event is not None:
            self._delete_circle_event(node.left_circle_event)
        if node.right_circle_event is not None:
            self._delete_circle_event(node.right_circle_event)

    def _split_beachline(self, direction, node, arc, event_face):
        #If site is directly below the arc, or on the right of the arc, add it as a successor
        if direction is Directions.CENTRE or direction is Directions.RIGHT:
            new_node = self.beachline.insert_successor(node, arc)
            duplicate_node = self.beachline.insert_successor(new_node, node.value)
        else:
            #otherwise add it as a predecessor
            new_node = self.beachline.insert_predecessor(node, arc)
            duplicate_node = self.beachline.insert_predecessor(new_node, node.value)
        assert(isinstance(new_node, Node))
        
        #add in the faces as a data point for the new node and duplicate
        new_node.data['face'] = event_face
        duplicate_node.data['face'] = node.data['face']

        #Debug the new triple: [ A, B, A]
        newTriple = [node.value.id,new_node.value.id,duplicate_node.value.id]
        tripleString = "-".join([ascii_uppercase[x % 26] for x in newTriple])
        logging.debug("Split {} into {}".format(str(newTriple[0]),tripleString))
        return new_node, duplicate_node


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
            left_circle = utils.math.get_circle_3p(*left_points)
            if left_circle and not utils.math.isClockwise(*left_points,cartesian=True):
                left_circle_loc = utils.math.get_lowest_point_on_circle(*left_circle)
                #check the l_t_p/s arent in this circle
                #note: swapped this to add on the right ftm
                self._add_circle_event(left_circle_loc,left_triple[1],left_circle[0],left=False)
            else:
                logging.debug("Left circle response: {}".format(left_circle))

        if right_triple:
            logging.debug("Calc Right Triple: {}".format("-".join([str(x) for x in right_triple])))
            
        if right and right_triple and right_triple[0].value != right_triple[2].value:
            right_points = [x.value.get_focus() for x in right_triple]
            right_circle = utils.math.get_circle_3p(*right_points)
            if right_circle and not utils.math.isClockwise(*right_points,cartesian=True):
                right_circle_loc = utils.math.get_lowest_point_on_circle(*right_circle)
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
        
        #get only the halfedges that are originless, rather than full edges that are infinite
        i_pairs = [x for x in self.halfEdges.items() if x[1].origin is None]
        logging.debug("Origin-less half edges num: {}".format(len(i_pairs)))
        
        #----
        #i_pairs = [((breakpoint nodes),edge)]
        for ((a,b),c) in i_pairs:
            i += 1
            #a and b are nodes
            logging.debug("{} Infinite Edge resolution: {}-{}, infinite? {}".format(i,a,b,c.isInfinite()))
            if not c.isInfinite():
                logging.debug("An edge that is not infinite")
                assert(False)
            #intersect the breakpoints to find the vertex point
            intersection = a.value.intersect(b.value)
            if intersection is None or intersection.shape[0] < 1:
                raise Exception("No intersections detected when completing an infinite edge")
            
            #Create a vertex for the end
            newVertex = self.dcel.newVertex(intersection[0,0],intersection[0,1])
            c.addVertex(newVertex)
            if c.isInfinite():
                raise Exception("After modification is infinite")

            
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
        """ Deactiate a circle event rather than deleting it.
        This means instead of removal and re-heapifying, you just skip the event
        when you come to process it """
        logging.debug("Deactivating Circle Event: {}".format(event))
        event.deactivate()
        
