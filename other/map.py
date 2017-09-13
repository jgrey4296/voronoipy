import dcel
import logging

class Map:

    def __init__(self,dcel,bbox):
        self.dcel = dcel
        self.bbox = bbox
        self.mapNodes = []

    def process_dcel(self):
        """ add map nodes to faces, annotate, etc  """
        faces = dcel.faces
        for face in faces:
            self.mapNodes.append(MapNode(face,self.bbox))

            
class MapNode:
    """ Date to store in a DCEL face, providing map related annotations for the face """
    
    def __init__(self,face,bbox):
        #Link face with map node
        self.face = face
        if self.face.data is not None:
            raise Exception("MapNode's face.data is not None")
        self.face.data = self

        #data of the map node:
        self.elevation = 0
        self.water = True
        self.mapEdge = False
        self.biome = None
        self.population = None
        self.neighbours = []

        #Run pre-processing:
        self.checkForMapEdge(bbox)
        
    def checkForMapEdge(self,bbox):
        nodeEdges = self.face.edgeList
        nodeVertices = [x.origin for x in nodeEdges]
        for vertex in nodeVertices:
            #not using vertex.within/outside because i'm looking for vertices ON
            #the bbox
            if vertex.x <= bbox[0] or vertex.x >= bbox[2]:
                self.mapEdge = True
            elif vertex.y <= bbox[1] or vertex.y >= bbox[3]:
                self.mapEdge = True



    
