* A Simple Python Fortune's Algorithm Voronoi Module

Not finding a decent tutorial on Fortune's Algorithm, 
I have implemented it from scratch. It isn't fast, but its on its 
way to clear enough to pick apart and understand. 

** Dependencies
Cairo, pyqtree

** Structure

Parabola and Quadratic define the base math required.
beachline is a modified rbtree for the purposes of fortune's alg.
dcel is a Double-Edge-Connected-List
voronoi is the main module
utils is for convenience functions, mainly to draw the dcel.
