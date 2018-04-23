""" Events: The data representations of points and circles 
    in the voronoi calculation
"""
import IPython
import numpy as np

class VEvent:
    offset = 0
    
    """ The Base Class of Events """
    def __init__(self,site_location,i=-1):
        assert(isinstance(site_location, np.ndarray))
        self.loc = site_location #tuple
        self.step = i

    def y(self):
        return self.loc[1]

    def __lt__(self,other):
        return (VEvent.offset - self.y()) < (VEvent.offset - other.y())
    
class SiteEvent(VEvent):
    """ Subclass for representing individual points / cell centres """
    def __init__(self,site_loc,i=None,face=None):
        super().__init__(site_loc,i=i)
        self.face = face
        
    def __str__(self):
        return "Site Event: Loc: {}".format(self.loc)

class CircleEvent(VEvent):
    """ Subclass for representing the lowest point of a circle, 
    calculated from three existing site events """
    def __init__(self,site_loc,sourceNode,voronoiVertex,left=True,i=None):
        if left and (sourceNode.right_circle_event is not None and sourceNode.right_circle_event.active):
            raise Exception("Trying to add a circle event to a taken left node: {} : {}".format(sourceNode,sourceNode.right_circle_event))
        elif not left and (sourceNode.left_circle_event is not None and sourceNode.left_circle_event.active):
            raise Exception("Trying to add a circle event to a taken right node: {} : {}".format(sourceNode,sourceNode.left_circle_event))
        super().__init__(site_loc,i=i)
        #The node that will disappear
        self.source = sourceNode
        #the breakpoint where it will disappear
        self.vertex = voronoiVertex #vertex == centre of circle, not lowest point
        self.active = True
        #is on the left
        self.left = left
        if left:
            sourceNode.right_circle_event = self
        else:
            sourceNode.left_circle_event = self
            
    def __str__(self):
        return "Circle Event: {}, Node: {}, Left: {}, Added On Step: {}".format(self.loc,
                                                                                self.source,
                                                                                self.left,
                                                                                self.step)
            
    def deactivate(self):
        """ Deactivating saves on having to reheapify """
        self.active = False

