import unittest
import logging
import numpy as np
from math import sin,cos,pi
from random import random
from test_context import dcel
from dcel import Vertex,HalfEdge,Face,DCEL
from manualcheck import checkVertices

TWO_PI = 2 * pi

class DCEL_HEdge_Sort_Tests(unittest.TestCase):

    def setUp(self):
        self.dcel = DCEL()
        self.face = self.dcel.newFace(0,0)
        #override the face bbox
        self.face.getCentroid = lambda: np.array([0.5,0.5])

    def tearDown(self):
        self.dcel = None
        self.face = None

    ### Testing that a < b == a is anti-cw of b
    ### Note: coordinate system is 0,0 in the top left corner
    
    def test_simple_anti_clockwise(self):
        #v1 is CW of v2
        v1 = Vertex(0.6,0.4)
        v2 = Vertex(0.6,0.6)
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="tsac")
        self.assertTrue(np.all(self.face.getCentroid() == np.array([0.5,0.5])))
        self.assertTrue(e1<e2)

    def test_simple_checking_coordinate_system(self):
        #v1 is CW of v2
        v1 = Vertex(0,1.0)
        v2 = Vertex(-0.3090169,0.9510565)
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        self.face.getCentroid = lambda: np.array([0.0,0.0])
        checkVertices(v1,v2,self.face.getCentroid(),testName="tsccs")
        self.assertTrue(np.all(self.face.getCentroid() == np.array([0.0,0.0])))
        self.assertGreater(e2,e1)
        self.assertLess(e1,e2)

    def test_simple_checking_coordinate_system_opp(self):
        #v2 is CCW of v1
        v1 = Vertex(sin(pi*0.1),cos(pi*0.1))
        v2 = Vertex(sin(pi*0.2),cos(pi*0.2))
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        self.face.getCentroid = lambda: np.array([0.0,0.0])
        checkVertices(v1,v2,self.face.getCentroid(),testName="tsccso")
        self.assertTrue(np.all(self.face.getCentroid() == np.array([0.0,0.0])))
        self.assertFalse(e1<e2)

        
    def test_anti_cw_3(self):
        #v2 is CCW of v1
        v1 = Vertex(1,0.5)
        v2 = Vertex(0.6,0.6)
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="tac3")
        self.assertTrue(e1<e2)

    def test_opposites(self):
        #v1 is CCW of v2
        v1 = Vertex(0.3,0.5)
        v2 = Vertex(1,0.5)
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="to")
        self.assertTrue(e1<e2)

    def test_wrapped(self):
        # v1 in this is CW from v2
        v1 = Vertex(0.4,0.4)
        v2 = Vertex(0.4,0.6)
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="tw")
        self.assertFalse(e1<e2)

    def test_scaled_up(self):
        """ Taken from voronoi error """
        #v1 is CW from v2
        v1 = Vertex(0.622,0.054)
        v2 = Vertex(0.431,0.027)
        self.face.getCentroid = lambda: np.array([0.55,0.08])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="tsu")
        self.assertFalse(e1<e2)

    def test_scaled_up_2(self):
        #v1 is CW from v2
        v1 = Vertex(0.431,0.027)
        v2 = Vertex(0.622,0.054)
        self.face.getCentroid = lambda: np.array([0.53,0.02])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="tsu2")
        self.assertFalse(e1<e2)
        
    def test_scaled_3(self):
        #v1 is CCW from v2
        v1 = Vertex(0.622,0.054)
        v2 = Vertex(0.664,0.013)
        self.face.getCentroid = lambda: np.array([0.53,0.02])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="ts3")
        self.assertFalse(e1<e2)

    def test_scaled_4(self):
        #v1 is CW from v2
        v1 = Vertex(0.664,0.013)
        v2 = Vertex(0.622,0.054)
        self.face.getCentroid = lambda: np.array([0.65,0.02])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="ts4")
        self.assertTrue(e1<e2)

    def test_scaled_5(self):
        #v1 is CW from v2
        v1 = Vertex(0.877,-0.0)
        v2 = Vertex(0.862,0.085)
        self.face.getCentroid = lambda: np.array([0.88,0.05])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="ts5")
        self.assertFalse(e1<e2)

    def test_scaled_6(self):
        #v1 is CW from v2
        v1 = Vertex(0.181,0.073)
        v2 = Vertex(0.110,0.0)
        self.face.getCentroid = lambda: np.array([0.09,0.07])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="ts6")
        self.assertFalse(e1<e2)

    def test_scaled_7(self):
        #v1 is CW from v2
        v1 = Vertex(0.670,0.0)
        v2 = Vertex(0.664,0.013)
        self.face.getCentroid = lambda: np.array([0.72,0.06])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="ts7")
        self.assertFalse(e1<e2)

    def test_scaled_8(self):
        #v1 is CCW from v2
        v1 = Vertex(0.669,0.390)
        v2 = Vertex(0.539,0.304)
        self.face.getCentroid = lambda: np.array([0.61,0.29])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="ts8")
        self.assertTrue(e1<e2)

    def test_scaled_9(self):
        #v1 is CW from v2
        v1 = Vertex(0.364,0.062)
        v2 = Vertex(0.289,0.168)
        self.face.getCentroid = lambda: np.array([0.32,0.14])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="ts9")
        self.assertFalse(e1<e2)


    def test_raw_voronoi(self):
        v1 = Vertex(0.201,0.0244)
        v2 = Vertex(0.206,0.0)
        self.face.getCentroid = lambda: np.array([0.107,0.001])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="trv")
        self.assertFalse(e1<e2)

    def test_raw_voronoi_2(self):
        v1 = Vertex(0.206,0.0)
        v2 = Vertex(0.201,0.0244)
        self.face.getCentroid = lambda: np.array([0.296,0.041])
        e1 = self.dcel.newEdge(v1,v2,self.face)
        e2 = self.dcel.newEdge(v2,v1,self.face)
        checkVertices(v1,v2,self.face.getCentroid(),testName="trv2")
        self.assertFalse(e1<e2)

        

        
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
