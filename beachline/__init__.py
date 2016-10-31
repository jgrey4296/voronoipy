## A Red-Black Tree bastardisation into a Beach Line for fortune's
# Properties of RBTrees:
# 1) Every node is Red Or Black
# 2) The root is black
# 3) Every leaf is Black, leaves are null nodes
# 4) If a node is red, it's children are black
# 5) All paths from a node to its leaves contain the same number of black nodes

from string import ascii_uppercase
import IPython
import numpy as np
import utils
import math
import logging



#--------------------
#def Beachline Container
#--------------------
class BeachLine(object):

    def __init__(self,arc=True):
        """ Initialise the rb tree container, ie: the node list """
        self.arc = arc
        self.nodes = []      #list of all nodes created 
        self.arcs_added = [] #list of all values added
        self.root = NilNode
        #init nilnode
        self.root.parent = NilNode
        self.root.left = NilNode
        self.root.right = NilNode
        self.root.canonical = NilNode
        
        
    def __str__(self):
        if self.root == NilNode:
            return "_"
        else:
            str = []
            str.append("RB Tree:\n")
            str.append("Colours: " + self.root.print_colour() + "\n")
            str.append("Heights: " + self.root.print_blackheight() + "\n")
            str.append("P_ids: " + self.root.print_tree() + "\n")
            str.append("Chain ids: " + self.print_chain())
            return "".join(str)
                
    def print_chain(self):
        if not self.arc:
            lst = [str(x) for x in self.get_chain()]
        else:
            lst = [ascii_uppercase[x.value.id] for x in self.get_chain()]
        return "-".join(lst)
        
    def update_arcs(self,d):
        for arc in self.arcs_added:
            arc.update_d(d)
        
    def isEmpty(self):
        if self.root == NilNode:
            return True
        return False

    def insert_many(self,*values):
        for x in values:
            self.insert(x)
    
    def insert(self,value):
        if self.root == NilNode:
            self.arcs_added.append(value)
            self.root = Node(value,arc=self.arc)
            self.nodes.append(self.root)
            self.balance(self.root)
            return self.root
        else:
            node,direction = self.search(value)
            if isinstance(direction,Right) or isinstance(direction,Centre):
                return self.insert_successor(node,value)
            else: #isinstance(direction,Left):
                return self.insert_predecessor(node,value)
        
    def insert_successor(self,existing_node,newValue):
        self.arcs_added.append(newValue)
        new_node = Node(newValue,arc=self.arc)
        self.nodes.append(new_node)
        if existing_node == NilNode:
            existing_node = self.root
        if existing_node == NilNode:
            self.root = new_node
        else:
            existing_node.add_right(new_node)
        self.balance(new_node)
        return new_node

    def insert_predecessor(self,existing_node,newValue):
        self.arcs_added.append(newValue)
        new_node = Node(newValue,arc=self.arc)
        self.nodes.append(new_node)
        if existing_node == NilNode :
            existing_node = self.root
        if existing_node == NilNode:
            self.root = new_node
        else:
            existing_node.add_left(new_node)
        self.balance(new_node)
        return new_node

    def delete_value(self,value):
        node,direction = self.search(value)
        self.delete_node(node)
    
    def delete_node(self,node):
        """ Delete a value from the tree """
        if node == NilNode:
            return        
        if self.arc:
            triple = [node.get_predecessor(),node,node.get_successor()]
            tripleString = "-".join([str(x) for x in triple if x])
            logging.info("Deleting Arc: {}".format(tripleString))
        else:
            logging.info("Deleting Value: {}".format(node.value))
        rbTreeDelete_textbook(self,node)
        if node in self.nodes:
            self.nodes.remove(node)
        #del node
        
    def search(self,x,d=None,verbose=False):
        """ Search the tree for a value, getting closest node to it, 
            returns (node,insertion_function)
        """
        current = self.root
        if current == NilNode:
            return None #not found
        parent = NilNode
        found = False
        while not found:
            comp = current.compare(x,d=d)
            logging.debug("Moving: {}".format(comp))
            if isinstance(comp,Left):
                parent = current
                current = current.left
            elif isinstance(comp,Right):
                parent = current
                current = current.right
            elif isinstance(comp,Centre):
                #ie: spot on
                parent = current
                found = True
            else: #type is none
                raise Exception("Comparison returned None")
            if current == NilNode:
                found = True
                
        return (parent, comp) #the existing parent and the side to add it
        
    def min(self):
        """ Get the min value of the tree """
        if self.root == NilNode:
            return None
        return self.root.getMin()

    def max(self):
        """ Get the max value of the tree """
        if self.root == NilNode:
            return None
        return self.root.getMax()
        
    def balance(self,node):
        rbtreeFixup(self,node)

    def get_chain(self):
        """ Get the sequence of values, from left to right """
        #logging.debug("getting chain of nodes")
        if self.root == NilNode:
            return []
        chain = []
        current = self.root.getMin()
        while current != NilNode:
            #logging.debug("Get_chain: appending {} Pre: {}, Succ:{}".format(current,current.predecessor,current.get_successor()))
            chain.append(current)
            current = current.get_successor()
        return [x for x in chain]

    def collapse_adjacent_arcs(self,node):
        """ If a node has the same arc as a successor/predecessor,
            just collapse them together, deleting one node
        """
        raise Exception
            
    def get_successor_triple(self,node):
        if node == NilNode:
            return None
        a = node
        b = a.get_successor()
        if b != NilNode:
            c = b.get_successor()
            if c != NilNode:
                return (a,b,c)
        return None

    def get_predecessor_triple(self,node):
        if node == NilNode:
            return None
        a = node
        b = a.get_predecessor()
        if b != NilNode:
            c = b.get_predecessor()
            if c != NilNode:
                return (c,b,a)
        return None
        
    def countBlackHeight(self,node=None):
        """ Given a node, count all paths and check they have the same black height """
        if node == NilNode:
            if self.root == NilNode:
                return None
        node = self.root
        stack = [node]
        leaves = []
        while len(stack) > 0:
            current = stack.pop()
            if current.isLeaf():
                leaves.append(current)
            else:
                if current.left != NilNode:
                    stack.append(current.left)
                if current.right != NilNode:
                    stack.append(current.right)
                        
        allHeights = [x.getBlackHeight(node) for x in leaves]
        return allHeights

#--------------------
#def Internal node
#--------------------

class NilNode(object):
    """ A Nil Node for use with the rb algorithms """
    canonical = False
    red = False
    parent = None
    right = None
    left = None
    arc = False
    value = None
    @staticmethod
    def __str__(self):
        return "Nil Node"

    @staticmethod
    def getMin():
        return NilNode.canonical

    @staticmethod
    def getMax():
        return NilNode.canonical

    @staticmethod
    def compare(x,d=None):
        raise Exception("Compare should not be called on NilNode")

    @staticmethod
    def compare_simple(x,d=None):
        raise Exception("Compare_simple should not be called on a NilNode")

    @staticmethod
    def isLeaf():
        return True

    @staticmethod
    def get_predecessor():
        return NilNode.canonical

    @staticmethod
    def get_successor():
        return NilNode.canonical

    @staticmethod
    def get_predecessor():
        return NilNode.canonical

    @staticmethod
    def get_successor():
        return NilNode.canonical

class Node(object):
    """ The internal node class for the rbtree.  """
    i = 0
    
    def __init__(self,value,parent=NilNode,red=True,arc=True):
        self.id = Node.i
        Node.i += 1
        #Node Data:
        self.red = red
        #arc: is the value an arc or a normal number?
        self.arc = arc
        self.value = value
        self.left_circle_event = None
        self.right_circle_event = None
        #Additional Data:
        self.data = {}
        #Children:
        self.left = NilNode
        self.right = NilNode
        #Parent:
        self.parent = parent

    def __str__(self):
        if self.arc:
            return ascii_uppercase[self.value.id]
        else:
            return str(self.value)

    def compare_simple(self,x):
        if x < self.value:
            return Left()
        if x > self.value:
            return Right()
        return Centre()
        
    def compare(self,x,d=None):
        logging.debug("Comparing {} to {}".format(x,self))
        if not self.arc:
            return self.compare_simple(x)
        pred = self.get_predecessor()
        succ = self.get_successor()
        logging.debug("Pred: {}, Succ: {}".format(pred,succ))
        pred_intersect = None
        succ_intersect = None
        the_range = [-math.inf,math.inf]
        if pred == NilNode and succ == NilNode: #Base case: single arc
            logging.debug("Single Arc: {}".format(self))
            return Centre()

        if pred != NilNode and succ != NilNode:
            logging.debug("Trio of arcs is clockwise: {}".format(utils.isClockwise(pred.value.get_focus(),\
                                                                           self.value.get_focus(),\
                                                                           succ.value.get_focus(),\
                                                                           cartesian=True)))
        #pred and successor are the same arc
        if pred != NilNode and succ != NilNode and pred.value == succ.value:
            intersect = pred.value.intersect(self.value)
            logging.warning("Predecessor and Successor are the same: {}".format(pred))
            logging.debug("Intersection result: {}".format(intersect))
            if len(intersect) != 2:
                raise Exception("Two parabolas arent intersecting correctly")
            if the_range[0] < intersect[:,0].min():
                the_range[0] = intersect[:,0].min()
            if the_range[1] > intersect[:,0].max():
                the_range[1] = intersect[:,0].max()
            

        else: #different arcs bookend
            if pred != NilNode:
                pred_intersect = self.value.intersect(pred.value)
                logging.debug("Pred intersect result: {}".format(pred_intersect))
                if len(pred_intersect) > 0:
                    the_range[0] = pred_intersect[1,0]
                    
            
            if succ != NilNode:
                succ_intersect = succ.value.intersect(self.value)
                logging.debug("Succ intersect result: {}".format(succ_intersect))
                if len(succ_intersect) > 0:
                    the_range[1] = succ_intersect[1,0]

        logging.debug("Testing: {} < {} < {}".format(the_range[0],x,the_range[1]))
        if the_range[0] < x and x <= the_range[1]:
            return Centre()
        elif x < the_range[0]:
            return Left()
        elif the_range[1] < x:
            return Right()
        else:
            raise Exception("Comparison failure")

        
    def isLeaf(self):
        return self.left == NilNode and self.right == NilNode
    
    def intersect(self,node):
        if not self.arc or not node.arc:
            raise Exception("Cant intersect on a non-arc node")
        else:
            raise Exception("Not implemented yet: intersect")

    def update_arcs(self,d):
        if not self.arc:
            raise Exception("Can't update arc on a non-arc node")
        self.arc.update_d(d)
        
    def getBlackHeight(self,root=NilNode):
        """ Get the number of black nodes from self to the root  """
        #logging.debug("Getting black height for {} to {}".format(self,root))
        current = self
        height = 0
        while current != root and current != NilNode:
            #logging.debug("looping: {} -> {}".format(current,current.parent))
            if not current.red:
                height += 1
            current = current.parent
        return height

    def countBlackHeight_null_add(self):
        """ Given a node, count all paths and check they have the same black height """
        stack = [self]
        leaves = []
        while len(stack) > 0:
            current = stack.pop()
            if current.isLeaf():
                leaves.append(current)
            else:
                if current.left != NilNode:
                    stack.append(current.left)
                if current.right != NilNode:
                    stack.append(current.right)

        #plus one for the true 'leaf' nodes, the nill ones
        allHeights = [x.getBlackHeight(self)+1 for x in leaves]
        return allHeights
    
    def print_colour(self):
        """ String representation of the node """
        #logging.debug("printing colours")
        if self.red:
            colour = "R"
        else:
            colour = "B"
        if self.isLeaf():
            return "{}".format(colour)
        else:
            a = None
            b = None
            if self.left != NilNode:
                a = self.left.print_colour()
            if self.right != NilNode:
                b = self.right.print_colour()
            return "{}( {} {} )".format(colour,a,b)

    def print_blackheight(self):
        #logging.debug("Printing heights")
        if self.isLeaf():
            return "{}".format(self.getBlackHeight())
        else:
            a = None
            b = None
            if self.left != NilNode:
                a = self.left.print_blackheight()
            if self.right != NilNode:
                b = self.right.print_blackheight()
            return "{}( {} {})".format(self.getBlackHeight(), a,b)
        
    def print_tree(self):
        #logging.debug("Printing tree")
        if not self.arc:
            return ""
        elif self.isLeaf():
            return ascii_uppercase[self.value.id]
        else:
            i = ascii_uppercase[self.value.id]
            a = "Nil"
            b = "Nil"
            if self.left != NilNode:
                a = self.left.print_tree()
            if self.right != NilNode:
                b = self.right.print_tree()
            return "{}( {} {} )".format(i,a,b)

    def print_tree_plus(self):
        #logging.debug("printing tree plus")
        if not self.arc:
            return ""
        elif self.isLeaf():
            p = str(self.get_predecessor())
            s = str(self.get_successor())
            return "{}<-{}->{}".format(p,ascii_uppercase[self.value.id],s)
        else:
            i = ascii_uppercase[self.value.id]
            p = str(self.get_predecessor())
            s = str(self.get_successor())
            a = "Nil"
            b = "Nil"
            if self.left != NilNode:
                a = self.left.print_tree_plus()
            if self.right != NilNode:
                b = self.right.print_tree_plus()
            return "{}<-{}->{}( {} {})".format(p,i,s,a,b)

        
    def get_predecessor(self):
        if self.left != NilNode:
            return self.left.getMax()
        current = self
        found = False
        while not found:
            #logging.debug("predecessor loop: {}".format(current))
            if current.parent.right == current or current.parent == NilNode:
                found = True
            current = current.parent
        #logging.debug("pred loop fin: {}".format(current))
        return current

    def get_successor(self):
        if self.right != NilNode:
            return self.right.getMin()
        current = self
        found = False
        while not found:
            #logging.debug("successor loop: {}".format(current))
            if current.parent.left == current or current.parent == NilNode:
                found = True
            current = current.parent
        #logging.debug("succ loop fin: {}".format(current))
        return current
        
    def getMin(self):
        """ Get the smallest leaf from the subtree this node is root of """
        current = self
        while not current.isLeaf() and current.left != NilNode:
            current = current.left
        return current
    
    def getMax(self):
        """ Get the largest leaf from the subtree this node is root of """
        current = self
        while not current.isLeaf() and current.right != NilNode:
            current = current.right
        return current
    
    def getMinValue(self):
        return self.getMin().value

    def getMaxValue(self):
        return self.getMax().value
    
    def add_left(self,node,force=False):
        logging.debug("{}: Adding {} to Left".format(self,node))
        if self == node:
            node = NilNode
        if self.left == NilNode or force:
            self.link_left(node)
        else:
            self.get_predecessor().add_right(node)
        
    def add_right(self,node,force=False):
        logging.debug("{}: Adding {} to Right".format(self,node))
        if self == node:
            node = NilNode
        if self.right == NilNode or force:
            self.link_right(node)
        else:
            self.get_successor().add_left(node)
        
    def disconnect_from_parent(self):
        if self.parent != NilNode:
            if self.parent.left == self:
                logging.debug("Disconnecting {} L-> {}".format(self.parent,self))
                self.parent.left = NilNode
            else:
                logging.debug("Disconnecting {} R-> {}".format(self.parent,self))
                self.parent.right = NilNode
            self.parent = None

    def link_left(self,node):
        logging.debug("{} L-> {}".format(self,node))
        if self == node:
            node = NilNode
        self.left = node
        self.left.parent = self

    def link_right(self,node):
        logging.debug("{} R-> {}".format(self,node))
        if self == node:
            node = NilNode
        self.right = node
        self.right.parent = self
            
    def disconnect_sequence(self):
        self.disconnect_successor()
        self.disconnect_predecessor()

    def disconnect_hierarchy(self):
        return [self.disconnect_left(),self.disconnect_right()]

    def disconnect_left(self):
        logging.debug("{} disconnectin left: {}".format(self,self.left))
        if self.left != NilNode:
            node = self.left
            self.left = NilNode
            node.parent = NilNode
            return node
        return None

    def disconnect_right(self):
        logging.debug("{} disconnecting right: {}".format(self,self.right))
        if self.right != NilNode:
            node = self.right
            self.right = NilNode
            node.parent = NilNode
            return node
        return None

            
#--------------------
# def Helper functions
#--------------------

def rotateLeft(tree,node):
    """ Rotate the given node left, making the new head be node.right """
    logging.debug("Rotating Left: {}".format(node))
    if node.right == NilNode or node == NilNode:
        return
        #raise Exception("Rotating left when there is no right")
    newHead = node.right #Get the right subtree
    originalParent = node.parent
    #left subtree becomes the right subtree:
    node.right = newHead.left
    if node.right != NilNode:
        node.right.parent = node
    #move the original node to the left
    newHead.left = node
    newHead.left.parent = newHead
    if originalParent == NilNode:            #update the root of the tree
        newHead.parent = NilNode
        if tree:
            tree.root = newHead
    elif node == originalParent.left:  #update the parent's left subtree
        originalParent.left = newHead
        newHead.parent = originalParent
    else:
        originalParent.right = newHead
        newHead.parent = originalParent
    return newHead

def rotateRight(tree,node):
    """ Rotate the given node right, making the new head be node.left """
    logging.debug("Rotating Right: {}".format(node))
    if node == NilNode:
        raise Exception("Rotating right when there is no left")
    newHead = node.left
    originalParent = node.parent
    node.left = newHead.right
    if node.left != NilNode:
        node.left.parent = node
    newHead.right = node
    newHead.right.parent = newHead    
    if originalParent == NilNode:
        newHead.parent = NilNode
        if tree:
            tree.root = newHead
    elif node == originalParent.left:
        originalParent.left = newHead
        newHead.parent = originalParent
    else:
        originalParent.right = newHead
        newHead.parent = originalParent
    return newHead
        
def rbtreeFixup(tree,node):
    """ Verify and fix the RB properties hold """
    while node.parent != NilNode and node.parent.red:
        parent = node.parent
        grandParent = parent.parent
        if grandParent == NilNode:
            break
        elif parent == grandParent.left:
            y = grandParent.right
            if y != NilNode and y.red:
                parent.red = False
                y.red = False
                grandParent.red = True
                node = grandParent
            else:
                if node == parent.right:
                    node = parent
                    rotateLeft(tree,node)#invalidates parent and grandparent
                node.parent.red = False
                node.parent.parent.red = True
                rotateRight(tree,node.parent.parent)
        else:
            y = grandParent.left
            if y != NilNode and y.red:
                parent.red = False
                y.red = False
                grandParent.red = True
                node = grandParent
            else:
                if node == parent.left:
                    node = parent
                    rotateRight(tree,node)#invalidates parent and grandparent
                node.parent.red = False
                node.parent.parent.red = True
                rotateLeft(tree,node.parent.parent)
    tree.root.red = False

def transplant(tree,u,v):
    """ Transplant the node v, and its subtree, in place of node u """
    logging.debug("Transplanting {} into {}".format(v,u))
    if u.parent == NilNode:
        logging.debug("Setting root to {}".format(v))
        tree.root = v
        v.parent = NilNode
    elif u == u.parent.left:
        logging.debug("Transplant linking left")
        parent = u.parent
        u.parent.link_left(v)
    else:
        logging.debug("Transplant linking right")
        parent = u.parent
        u.parent.link_right(v)


def rbTreeDelete_textbook(tree,z):
    y = z
    orig_pred = z.get_predecessor()
    orig_succ = z.get_successor()
    orig_parent = z.parent
    y_originally_red = y.red
    x = NilNode
    if z.left == NilNode:
        logging.debug("No left, transplanting right")
        x = z.right
        transplant(tree,z,z.right)
    elif z.right == NilNode:
        logging.debug("No right, transplanting left")
        x = z.left
        transplant(tree,z,z.left)
    else:
        logging.debug("Both Left and right exist")
        y = z.right.getMin()
        y_originally_red = y.red
        x = y.right
        if y.parent == z:
            logging.debug("y.parent == z")
            x.parent = y
        else:
            logging.debug("y.parent != z")
            transplant(tree,y,y.right)
            y.link_right(z.right)
        transplant(tree,z,y)
        y.link_left(z.left)
        y.red = z.red
    if not y_originally_red:
        logging.debug("Fixingup up x: {}".format(x))
        rbDeleteFixup_textbook(tree,x)
    #collapse when two nodes are the same
    if orig_pred != NilNode and orig_succ != NilNode and orig_pred.value == orig_succ.value:
        logging.info("Collapsing with successor {}".format(orig_succ))
        tree.delete(orig_succ)
    logging.debug("Finished deletion")
    
def rbDeleteFixup_textbook(tree,x):
    while x != tree.root and not x.red:
        if x == x.parent.left:
            w = x.parent.right
            if w.red:
                w.red = False
                x.parent.red = True
                rotateLeft(tree,x.parent)
                w = x.parent.right
            if not w.left.red and not w.right.red:
                w.red = True
                x = x.parent
            else:
                if not w.right.red:
                    w.left.red = False
                    w.red = True
                    rotateRight(tree,w)
                    w = x.parent.right
                w.red = x.parent.red
                x.parent.red = False
                w.right.red = False
                rotateLeft(tree,x.parent)
                x = tree.root
        else: #mirror for right
            w = x.parent.left
            if w.red:
                w.red = False
                x.parent.red = True
                rotateRight(tree,x.parent)
                w = x.parent.left
            if not w.right.red and not w.left.red:
                w.red = True
                x = x.parent
            else:
                if not w.left.red:
                    w.right.red = False
                    w.red = True
                    rotateLeft(tree,w)
                    w = x.parent.left
                w.red = x.parent.red
                x.parent.red = False
                w.left.red = False
                rotateRight(tree,x.parent)
                x = tree.root
    x.red = False
 

#--------------------
# def UTILITY DIRECTION OBJECTS:
#--------------------
class Direction(object):
    def __init__(self):
        return
    def __str__(self):
        return "Direction"
class Left(Direction):
    def __init__(self):
        return
    def __str__(self):
        return "Left"
class Right(Direction):
    def __init__(self):
        return
    def __str__(self):
        return "Right"
class Centre(Direction):
    def __init__(self):
        return
    def __str__(self):
        return "Centre"


