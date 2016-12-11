import os
import sys
sys.path.insert(0,os.path.abspath('..'))
from math import pi,cos,sin
import cairo
import utils
from os.path import join

imgPath = "../imgs"
imgName = "check"

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,1000,1000)
ctx = cairo.Context(surface)
ctx.scale(1000,1000)
ctx.set_font_size(0.03)

def checkVertices(v1,v2,c,testName=None):
    check(v1.x,v1.y,v2.x,v2.y,c[0],c[1],testName=testName)

def check(x1,y1,x2,y2,cx,cy,testName=None):
    """
    Given two points and a centre, draw the image
    """
    utils.clear_canvas(ctx)
    #P1: Red
    ctx.set_source_rgba(1,0,0,1)
    utils.drawCircle(ctx,x1,y1,0.01)
    #P2: Green
    ctx.set_source_rgba(0,1,0,1)
    utils.drawCircle(ctx,x2,y2,0.01)
    #C: Blue
    ctx.set_source_rgba(0,0,1,1)
    utils.drawCircle(ctx,cx,cy,0.01)
    if testName:
        testImageName = "{}_{}".format(imgName,testName)
    else:
        testImageName = imgName
    utils.write_to_png(surface,join(imgPath,testImageName))
