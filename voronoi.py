import numpy as np
import numpy.random as random
import pyqtree
import IPython
import heapq
import pickle
import logging
import sys
from os.path import isfile
from string import ascii_uppercase
from numpy.linalg import det
from math import pi, sin, cos, nan

from beachline import BeachLine, NilNode, Node, Parabola
from beachline.utils import Directions
from dcel import DCEL, utils

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
currentStep = 0
EPSILON = sys.float_info.epsilon

class Voronoi(object):
    """ Creates a random selection of points, and step by step constructs
        a voronoi diagram
    """
    def __init__(self,ctx,sizeTuple,num_of_nodes=10):
        self.ctx = ctx
        self.sX = sizeTuple[0]
        self.sY = sizeTuple[1]
        self.nodeSize = num_of_nodes
        #Heap of site/circle events
        self.events = []
        #backup of the original sites
        self.sites = []
        #backup of all circle events
        self.circles = []

        self.beachline = None
        #The sweep line position
        self.sweep_position = None
        #The output voronoi diagram as a DCEL
        self.dcel = None
        #storage of breakpoint tuples -> halfedge
        self.halfEdges = {}

    def initGraph(self,data=None,rerun=False):
        """ Create a graph of initial random sites """
        logging.debug("Initialising graph")
        self.dcel = DCEL(bbox=BBOX) #init the dcel
        self.events = []
        self.sites = []
        self.circles = []
        self.halfEdges = {}
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
        
        #Create beachline
        self.beachline = BeachLine()

        #Save the nodes
        if not rerun:
            self.save_graph(values)
        return values

    def relax(self):
        """ Having calculated the voronoi diagram, use the centroids of 
            the faces instead of the sites, and retrun the calculation
        """
        if len(self.events) != 0:
            raise Exception("Calculation incomplete")
        newSites = [x.getAvgCentroid() for x in self.dcel.faces]
        logging.info("Num of Faces: {}".format(len(newSites)))
        self.initGraph(data=newSites,rerun=True)
        self.calculate_to_completion()

        
    def calculate_to_completion(self):
        finished = False
        i = 0
        while not finished:
            logging.debug("Calculating step: {}".format(i))
            finished = self._calculate(i)
            i += 1

    def _calculate(self,i):
        """ Calculate the next step of the voronoi diagram,
            Return True on completion, False otherwise
        """
        global currentStep
        currentStep = i
        if len(self.events) == 0: #finished calculating
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
        if len(self.events) != 0:
            raise Exception("Voronoi calculation not completed")
        logging.debug("\n\nFinalising DCEL")
        logging.debug(self.dcel)
        #Not a pure DCEL operation as it requires curve intersection:
        self._complete_edges()
        #pure DCEL operations:
        self.dcel.purge_infinite_edges()
        self.dcel.constrain_half_edges(bbox=BBOX)
        self.dcel.fixup_halfedges()
        #todo: connect edges of faces together
        self._complete_faces()
        #logging.debug(self.dcel)
        return self.dcel

    #FORTUNE METHODS
    def _handleSiteEvent(self,event):
        """
        provided with a site event, add it to the beachline in the appropriate place
        then update/remove any circle events that trios of arcs generate
        """
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

        if not isinstance(new_node,Node):
            raise Exception("Bad Creation of Beach line node")
            
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
        with open(SAVENAME,'wb') as f:
            pickle.dump(values,f)
        
    def load_graph(self):
        if isfile(SAVENAME):
            with open(SAVENAME,'rb') as f:
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
            event = CircleEvent(loc,sourceNode,voronoiVertex,i=currentStep)
        else:
            event = CircleEvent(loc,sourceNode,voronoiVertex,left=False,i=currentStep)
        logging.debug("Adding: {}".format(event))
        heapq.heappush(self.events,event)
        self.circles.append(event)

    def _delete_circle_event(self,event):
        logging.debug("Deleting Circle Event: {}".format(event))
        #self.events = [e for e in self.events if not e.nodeIs(event.source)]
        #heapq.heapify(self.events)
        event.deactivate()
        
    #-------------------- DEBUG Drawing Methods
    def draw_voronoi_diagram(self,clear=True):
        """ Draw the final diagram """
        logging.debug("Drawing final voronoi diagram")
        if clear:
            utils.clear_canvas(self.ctx)
        self.ctx.set_source_rgba(*COLOUR)
        #draw sites
        for site in self.sites:
            utils.drawCircle(self.ctx,*site.loc,0.007)
        #draw faces
        utils.drawDCEL(self.ctx,self.dcel)                       

    def draw_intermediate_states(self):
        logging.debug("Drawing intermediate state")
        utils.clear_canvas(self.ctx)
        self.draw_sites()
        self.draw_beach_line_components()
        self.draw_sweep_line()
        self.draw_circle_events()
        utils.drawDCEL(self.ctx,self.dcel)
        
    def draw_sites(self):
        self.ctx.set_source_rgba(*SITE_COLOUR)
        for site in self.sites:
            utils.drawCircle(self.ctx,*site.loc,SITE_RADIUS)

    def draw_circle_events(self):
        for event in self.circles:
            if event.active:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR)
                utils.drawCircle(self.ctx,*event.loc,CIRCLE_RADIUS)
            else:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR_INACTIVE)
                utils.drawCircle(self.ctx,*event.loc,CIRCLE_RADIUS)
                
    def draw_beach_line_components(self):
        #the arcs themselves
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR,0.1)
        xs = np.linspace(0,1,2000)
        for arc in self.beachline.arcs_added:
            xys = arc(xs)
            for x,y in xys:
                utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
        #--------------------
        #the frontier:
        # Essentially a horizontal travelling sweep line to draw segments
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
        leftmost_x = nan
        ##Get the chain of arcs:
        chain = self.beachline.get_chain()
        if len(chain) > 1:
            enumerated = list(enumerate(chain))
            pairs = zip(enumerated[0:-1],enumerated[1:])
            for (i,a),(j,b) in pairs:
                logging.debug("Drawing {} -> {}".format(a,b))
                intersections = a.value.intersect(b.value,self.sweep_position.y())
                #print("Intersections: ",intersections)
                if len(intersections) == 0:
                    logging.exception("NO INTERSECTION: {} - {}".format(i,j))
                    #Draw the non-intersecting line as red
                    self.ctx.set_source_rgba(*BEACH_NO_INTERSECT_COLOUR)
                    xs = np.linspace(0,1,2000)
                    axys = a.value(xs)
                    bxys = b.value(xs)
                    for x,y in axys:
                        utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
                    for x,y in bxys:
                        utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
                    self.ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
                    continue
                    #----------
                    #raise Exception("No intersection point")
                #intersection xs:
                i_xs = intersections[:,0]
                #xs that are further right than what we've drawn
                if leftmost_x is nan:
                    valid_xs = i_xs
                else:
                    valid_xs = i_xs[i_xs>leftmost_x]
                if len(valid_xs) == 0:
                    #nothing valid, try the rest of the arcs
                    continue
                left_most_intersection = valid_xs.min()
                logging.debug("Arc {0} from {1:.2f} to {2:.2f}".format(i,leftmost_x,left_most_intersection))
                if leftmost_x is nan:
                    leftmost_x = left_most_intersection - 1
                xs = np.linspace(leftmost_x,left_most_intersection,2000)
                #update the position
                leftmost_x = left_most_intersection
                frontier_arc = a.value(xs)
                for x,y in frontier_arc:
                    utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)

        if len(chain) > 0 and (leftmost_x is nan or leftmost_x < 1.0):
            if leftmost_x is nan:
                leftmost_x = 0
            #draw the last arc:
            logging.debug("Final Arc from {0:.2f} to {1:.2f}".format(leftmost_x,1.0))
            xs = np.linspace(leftmost_x,1.0,2000)
            frontier_arc = chain[-1].value(xs)
            for x,y in frontier_arc:
                utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
            
    def draw_sweep_line(self):
        if self.sweep_position is None:
            return        
        self.ctx.set_source_rgba(*SWEEP_LINE_COLOUR)
        self.ctx.set_line_width(LINE_WIDTH)
        #a tuple
        sweep_event = self.sweep_position
        self.ctx.move_to(0.0,sweep_event.y())
        self.ctx.line_to(1.0,sweep_event.y())
        self.ctx.close_path()
        self.ctx.stroke()


            

#--------------------
#Event class - For CIRCLE/SITE events

class VEvent:

    def __init__(self,site_location,i=-1):
        self.loc = site_location #tuple
        self.step = i

    def y(self):
        return self.loc[1]

    def __lt__(self,other):
        return self.y() < other.y()

    def nodeIs(self,other):
        return False
    
class SiteEvent(VEvent):
    def __init__(self,site_loc,i=None,face=None):
        super().__init__(site_loc,i=i)
        self.face = face
        
    def __str__(self):
        return "Site Event: Loc: {}".format(self.loc)

class CircleEvent(VEvent):
    def __init__(self,site_loc,sourceNode,voronoiVertex,left=True,i=None):
        if left and sourceNode.left_circle_event is not None:
            raise Exception("Trying to add a circle event to a taken left node: {} : {}".format(sourceNode,sourceNode.left_circle_event))
        elif not left and sourceNode.right_circle_event is not None:
            raise Exception("Trying to add a circle event to a taken right node: {} : {}".format(sourceNode,sourceNode.right_circle_event))
        super().__init__(site_loc,i=i)
        self.source = sourceNode
        self.vertex = voronoiVertex #vertex == centre of circle, not lowest point
        self.active = True
        self.left = left
        if left:
            sourceNode.left_circle_event = self
        else:
            sourceNode.right_circle_event = self
            
    def __str__(self):
        return "Circle Event: {}, Node: {}, Left: {}, Added On Step: {}".format(self.loc,
                                                                                self.source,
                                                                                self.left,
                                                                                self.step)
            
    def deactivate(self):
        self.active = False
        if self.left:
            self.source.left_circle_event = None
        else:
            self.source.right_circle_event = None
        #self.source = None

    def nodeIs(self,node):
        return self.source == node
