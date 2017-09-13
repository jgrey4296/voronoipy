import numpy as np
import numpy.random as random
from numpy.linalg import det
import math
from math import pi, sin, cos
import math
import pyqtree
import utils
import IPython
import heapq


from Tree import Tree
from Parabola import Parabola

import dcel
TWOPI = 2 * pi

COLOUR = [0.2,0.1,0.6,1.0]
COLOUR_TWO = [1.0,0.2,0.4,0.5]



class VExperiment(object):

    def __init__(self,ctx,sizeTuple,num_of_nodes):
        self.ctx = ctx
        self.sX = sizeTuple[0]
        self.sY = sizeTuple[1]
        self.nodeSize = num_of_nodes
        self.nodes = None
        
        #Nodes: Array of n horizontal and vertical lines
        self.graph = np.zeros((1,4))
        self.intersections = np.zeros((1,2))
        
    def initGraph(self):
        #insert a bunch of horizontal or vertical lines
        for x in range(self.nodeSize):
            choice = random.choice(['h','v'])
            if choice == 'h':
                self.graph = np.row_stack((self.graph,makeHorizontalLine()))
            else:
                self.graph = np.row_stack((self.graph,makeVerticalLine()))

    def calculate_lines(self):
        """ A Sweep line means of drawing intersections of horizontal and vertical lines """
        tree = Tree(0.5)
        active = []
        #separate into events
        events = self.graphToEvents()
        
        #go through each event:
        for e in events:
            if len(e) == 3:
                if e[-1] not in active:
                    ## if horizontal start - add to tree
                    active.append(e[-1])
                    tree.insert(e[1],data=e[-1])
                else:
                    ## if horizontal end - remove from tree
                    active.remove(e[-1])
                    v = tree.search(e[1])
                    if v is not None:
                        v.data = None
                    #tree.delete(e[1])
            elif len(e) == 4:
                ## if vertical - get range then search, and store intersections
                r = tree.getRange(e[1],e[2])
                #todo: mark intersections
                line_indices = [x.data for x in r if x.data is not None]
                crossPoints = [(e[0],d.value) for d in r if d.data is not None]
                for xy in crossPoints:
                    self.intersections = np.row_stack((self.intersections,xy))

    def draw_line_intersections(self):
        """ Draw the lines calculated by the above sweep line algorithm """
        #DRAW LINES
        self.ctx.set_source_rgba(*COLOUR)
        for (x,y,x2,y2) in self.graph:
            line = utils.createLine(x,y,x2,y2,1000)
            for x,y in line:
                utils.drawCircle(self.ctx,x,y,0.002)

        #DRAW INTERSECTIONS:
        self.ctx.set_source_rgba(*COLOUR_TWO)
        for (x,y) in self.intersections:
            utils.drawCircle(self.ctx,x,y,0.009)
 
    def drawTest_old(self):
        """ a grab bag of tests"""
        self.ctx.set_source_rgba(*COLOUR)
        # p = [0.3,0.6]
        # l = 0.9

        # utils.drawCircle(self.ctx,p[0],p[1],0.005)
        
        # line = utils.createLine(0,l,1,l,1000)
        # for x,y in line:
        #     utils.drawCircle(self.ctx,x,y,0.002)

        # par = makeParabola(p,l,np.linspace(0,1,1000))
        # print(par)
        # for x,y in par:
        #     utils.drawCircle(self.ctx,x,y,0.002)

        #DCEL Test:
        dc = dcel.DCEL()
        v1 = dc.newVertex(0.2,0.2)
        v2 = dc.newVertex(0.4,0.2)
        v3 = dc.newVertex(0.5,0.6)
        e1 = dc.newEdge(v1,v2)
        e2 = dc.newEdge(v2,v3)
        e3 = dc.newEdge(v3,v1)
        f1 = dc.newFace()
        dc.linkEdgesTogether([e1,e2,e3])
        dc.setFaceForEdgeLoop(f1,e1)
        #utils.drawDCEL(self.ctx,dc)

        #Draw an arc:
        centre = np.array([[0.5,0.5]])
        r = 0.2
        rads = pi/2
        self.ctx.set_line_width(0.002)
        for xy in centre:
            self.ctx.arc(*xy,r,0,rads)
        self.ctx.stroke()

        #Draw the end points
        p1 = centre + [r,0] 
        p2 = utils.rotatePoint(p1,centre,rads)
        utils.drawCircle(self.ctx,*centre[0],0.006)
        utils.drawCircle(self.ctx,*p1[0],0.006)
        utils.drawCircle(self.ctx,*p2[0],0.006)
        #draw a chord:
        self.ctx.move_to(*p1[0])
        self.ctx.line_to(*p2[0])
        self.ctx.stroke()
        
        #midpoint:
        e = 1 #direction 1/-1 counter/clockwise
        d = utils.get_distance(p1,p2)
        m = utils.get_midpoint(p1,p2)
        utils.drawCircle(self.ctx,*m[0],0.006)
        #normal:
        n = utils.get_normal(p1,p2)
        bisector = utils.get_bisector(p1,p2)
        h = np.sqrt(pow(r,2) - (pow(d,2) / 4)) #height from centre
        c = m + (e * h * bisector) #centre calculated
        self.ctx.move_to(*m[0])
        self.ctx.line_to(*c[0])
        self.ctx.stroke()

        #extend a line:
        self.ctx.set_source_rgba(0.8,0.6,0.2,1)
        extend_point = p2
        extend_distance = 5
        extended_line = utils.extend_line(p1,p2,extend_distance)
        self.ctx.move_to(*extend_point[0])
        self.ctx.line_to(*extended_line[0])
        self.ctx.stroke()

        #clip the line
        self.ctx.set_source_rgba(0.5,0.1,0.1,1)
        clippedLine = utils.bound_line_in_bbox(np.row_stack((extend_point,extended_line)),
                                                 [0.5,0,1,1])
        self.ctx.move_to(*clippedLine[0])
        self.ctx.line_to(*clippedLine[1])
        self.ctx.stroke()

        #intersect two lines 
        l1 = utils.random_points(2)
        l2 = utils.random_points(2)
        #l1 = np.array([0.2,0.2,0.4,0.4])
        #l2 = np.array([0.4,0.2,0.2,0.4])
        self.ctx.move_to(l1[0],l1[1])
        self.ctx.line_to(l1[2],l1[3])
        self.ctx.stroke()
        self.ctx.move_to(l2[0],l2[1])
        self.ctx.line_to(l2[2],l2[3])
        self.ctx.stroke()

        intersect = utils.intersect(l1,l2)
        print(intersect)
        if intersect is not None:
            utils.drawCircle(self.ctx,*intersect,0.005)


    def drawTest(self):
        #get two points and a sweep line position:
        a = np.random.random(2)
        b = np.random.random(2)
        c = max(a[1],b[1]) + np.random.random(1)
        
        self.ctx.set_source_rgba(1.0,0,0,1)
        utils.drawCircle(self.ctx,a[0],a[1],0.004)
        utils.drawCircle(self.ctx,b[0],b[1],0.004)
        
        self.ctx.set_line_width(0.002)
        self.ctx.move_to(0,c)
        self.ctx.line_to(1,c)
        self.ctx.stroke()

        p1 = Parabola(*a,c)
        p2 = Parabola(*b,c)

        #sample and draw:
        xs = np.linspace(0,1,1000)
        p1_xys = p1.calc(xs)
        p2_xys = p2.calc(xs)
        
        for x,y in p1_xys:
            utils.drawCircle(self.ctx,x,y,0.002)
        for x,y in p2_xys:
            utils.drawCircle(self.ctx,x,y,0.002)

        #intersect:
        intersections = p1.intersect(p2)
        print("intersections",intersections)
        self.ctx.set_source_rgba(0.0,1.0,0.0,1)
        for x,y in intersections:
            utils.drawCircle(self.ctx,x,y,0.004)
        

    def drawTest_old2(self):
        self.ctx.set_source_rgba(*COLOUR)

        #Draw three points
        base = np.random.random((3,2))
        baseS = np.column_stack((np.sort(base[:,0]),np.sort(base[:,1])))
            
        p1 = np.array([baseS[0]])
        p2 = np.array([baseS[1]])
        p3 = np.array([baseS[2]])

        utils.drawCircle(self.ctx,*p1[0],0.006)
        utils.drawCircle(self.ctx,*p2[0],0.006)
        utils.drawCircle(self.ctx,*p3[0],0.006)
        
        #connect them as two chords
        self.ctx.set_line_width(0.002)
        self.ctx.move_to(*p1[0])
        self.ctx.line_to(*p2[0])
        self.ctx.line_to(*p3[0])
        self.ctx.stroke()
        
        #Draw the normals from the midpoints
        arbitrary_height = 200
        m1 = utils.get_midpoint(p1,p2)
        n1 = utils.get_bisector(p1,p2,r=True)

        m2 = utils.get_midpoint(p2,p3)        
        n2 = utils.get_bisector(p2,p3,r=True)

        utils.drawCircle(self.ctx,*m1[0],0.007)
        utils.drawCircle(self.ctx,*m2[0],0.007)

        v1 = m1 + (1 * arbitrary_height * n1)
        v2 = m2 + (1 * arbitrary_height * n2)
        v1I = m1 + (-1 * arbitrary_height * n1)
        v2I = m2 + (-1 * arbitrary_height * n2)
        
        self.ctx.move_to(*m1[0])
        self.ctx.line_to(*v1[0])
        self.ctx.stroke()
        self.ctx.move_to(*m2[0])
        self.ctx.line_to(*v2[0])
        self.ctx.stroke()
        
        #intersect the two lines
        l1 = np.column_stack((m1,v1))
        l2 = np.column_stack((m2,v2))
        intersection = utils.intersect(l1[0],l2[0])
        #thus getting a centre point
        if intersection is None:
            #try inverted if not intersection
            print('trying inverted')
            l1i = np.column_stack((m1,v1I))
            l2i = np.column_stack((m2,v2I))
            intersection = utils.intersect(l1i[0],l2i[0])

        if intersection is not None:
            utils.drawCircle(self.ctx,*intersection,0.007)

            #Draw circles that go through the original points:
            #first get the radi
            r1 = utils.get_distance(p1,intersection)
            r2 = utils.get_distance(p2,intersection)
            r3 = utils.get_distance(p3,intersection)

            # self.ctx.arc(*intersection,r1,0,TWOPI)
            # self.ctx.arc(*intersection,r2,0,TWOPI)
            # self.ctx.arc(*intersection,r3,0,TWOPI)
            # self.ctx.stroke()

        #the above factored out into:
        icirc = utils.get_circle_3p(p1,p2,p3)
        if icirc is not None:
            self.ctx.arc(*icirc[0],icirc[1],0,TWOPI)
            self.ctx.stroke()
        else:
            print('No circle')
            

#--------------------
    def graphToEvents(self):
        #return lines turned into events
        events = []
        for i,(x,y,x2,y2) in enumerate(self.graph):
            if x == x2: #vertical
                events.append((x,y,y2,i))
            elif y == y2: #horizontal
                events.append((x,y,i))
                events.append((x2,y,i))
        return sorted(events)

#--------------------

def makeHorizontalLine():
    """ Describe a horizontal line as a vector of start and end points  """
    x = random.random()
    x2 = random.random()
    y = random.random()
    if x < x2:    
        return np.array([x,y,x2,y])
    else:
        return np.array([x2,y,x,y])


def makeVerticalLine():
    """ Describe a vertical line as a vector of start and end points """
    x = random.random()
    y = random.random()
    y2 = random.random()
    if y < y2:
        return np.array([x,y,x,y2])
    else:
        return np.array([x,y2,x,y])

def makeParabola(focus,directrix,xs):
    """ Return the xy coords for a given range of xs, with a focus point and bounding line """
    firstConst = 1 / (2 * ( focus[1] - directrix))
    secondConst = (focus[1] + directrix) / 2
    ys = firstConst * pow((xs - focus[0]),2) + secondConst
    xys = np.column_stack((xs,ys))
    return xys
    
    
#----------

def isSiteEvent(e):
    return True

def handleSiteEvent(event):
    return None

def handleCircleEvent(event):
    return None
