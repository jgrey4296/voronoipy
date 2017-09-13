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
