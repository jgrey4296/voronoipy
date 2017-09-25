import cairo_utils as utils
from os.path import join, isfile, exists, isdir, splitext, expanduser
from os import listdir
import numpy as np
from math import nan
import logging as root_logger
import IPython
logging = root_logger.getLogger(__name__)

#Constants:
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

class Voronoi_Debug:
    """ Contains methods to draw various stages of the voronoi calculation """

    def __init__(self, n, directory, instance):
        surf, ctx, size, n = utils.drawing.setup_cairo(N=n)
        assert(isdir(directory))
        self.instance = instance
        self.surface = surf
        self.ctx = ctx
        self.draw_size = size
        self.draw_n = n
        self.save_dir = directory


    def draw_voronoi_diagram(self,
                             i=None,
                             clear=True,
                             faces=True,
                             edges=False,
                             verts=False,
                             text=False):
        """ Draw the final diagram """
        logging.debug("Drawing final voronoi diagram")
        dcel = self.instance.finalise_DCEL()
        if clear:
            utils.clear_canvas(self.ctx)
        self.ctx.set_source_rgba(*COLOUR)
        #draw sites
        for site in self.instance.sites:
            utils.drawCircle(self.ctx, *site.loc, 0.007)
        #draw faces
        utils.drawDCEL(self.ctx, dcel, text=text, faces=faces, edges=edges, verts=verts)
        utils.write_to_png(self.surface, join(self.save_dir, "voronoi_debug"))
        

    def draw_intermediate_states(self, i,
                                 sites=True,
                                 beachline=True,
                                 sweepline=True,
                                 circles=True,
                                 dcel=False):
        """ Top level function to draw intermediate state of the algorithm """
        logging.info("Drawing intermediate state: {}".format(i))
        utils.clear_canvas(self.ctx)
        if sites:
            self.draw_sites()
        if beachline:
            self.draw_beach_line_components()
        if sweepline:
            self.draw_sweep_line()
        if circles:
            self.draw_circle_events()
        if dcel:
            #TODO: draw the incomplete lines better
            #backup_dcel_data = self.instance.dcel.export_data()
            #dcelInstance = self.instance.finalise_DCEL()
            utils.drawDCEL(self.ctx, self.instance.dcel)
            #self.instance.dcel.import_data(backup_dcel_data)
        utils.write_to_png(self.surface, join(self.save_dir, "voronoi_intermediate"), i=i)

        
    def draw_sites(self):
        self.ctx.set_source_rgba(*SITE_COLOUR)
        for site in self.instance.sites:
            utils.drawCircle(self.ctx, *site.loc, SITE_RADIUS)

    def draw_circle_events(self):
        for event in self.instance.circles:
            if event.active:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR)
                utils.drawCircle(self.ctx, *event.loc, CIRCLE_RADIUS)
            else:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR_INACTIVE)
                utils.drawCircle(self.ctx, *event.loc, CIRCLE_RADIUS)
                
    def draw_beach_line_components(self):
        #the arcs themselves
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR, 0.1)
        xs = np.linspace(0,1,2000)
        for arc in self.instance.beachline.arcs_added:
            xys = arc(xs)
            for x,y in xys:
                utils.drawCircle(self.ctx, x, y, BEACH_RADIUS)
        #--------------------
        #the frontier:
        # Essentially a horizontal travelling sweep line to draw segments
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
        leftmost_x = nan
        ##Get the chain of arcs:
        chain = self.instance.beachline.get_chain()
        if len(chain) > 1:
            enumerated = list(enumerate(chain))
            pairs = zip(enumerated[0:-1],enumerated[1:])
            for (i,a),(j,b) in pairs:
                logging.debug("Drawing {} -> {}".format(a,b))
                intersections = a.value.intersect(b.value, self.instance.sweep_position.y())
                logging.debug("Intersections: ",intersections)
                if len(intersections) == 0:
                    logging.exception("NO INTERSECTION: {} - {}".format(i,j))
                    #Draw the non-intersecting line as red
                    self.ctx.set_source_rgba(*BEACH_NO_INTERSECT_COLOUR)
                    xs = np.linspace(0, 1.0, 2000)
                    axys = a.value(xs)
                    bxys = b.value(xs)
                    for x,y in axys:
                        utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
                    for x,y in bxys:
                        utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
                    self.ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
                    continue
                #----------
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
                xs = np.linspace(leftmost_x, left_most_intersection, 2000)
                #update the position
                leftmost_x = left_most_intersection
                frontier_arc = a.value(xs)
                for x,y in frontier_arc:
                    utils.drawCircle(self.ctx, x, y, BEACH_RADIUS)

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
        if self.instance.sweep_position is None:
            return
        self.ctx.set_source_rgba(*SWEEP_LINE_COLOUR)
        self.ctx.set_line_width(LINE_WIDTH)
        #a tuple
        sweep_event = self.instance.sweep_position
        self.ctx.move_to(0.0,sweep_event.y())
        self.ctx.line_to(1.0,sweep_event.y())
        self.ctx.close_path()
        self.ctx.stroke()
