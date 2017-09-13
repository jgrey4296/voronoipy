
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

#-------------------- DEBUG Drawing Methods
def draw_voronoi_diagram(ctx, voronoi_instance,clear=True):
    """ Draw the final diagram """
    logging.debug("Drawing final voronoi diagram")
    dcel = voronoi_instance.finalise_DCEL()
    if clear:
        utils.clear_canvas(ctx)
    ctx.set_source_rgba(*COLOUR)
    #draw sites
    for site in voronoi_instance.sites:
        utils.drawCircle(ctx, *site.loc, 0.007)
    #draw faces
    utils.drawDCEL(ctx, dcel)                       

def draw_intermediate_states(ctx, voronoi_instance):
    """ Top level function to draw intermediate state of the algorithm """
    logging.debug("Drawing intermediate state")
    dcel = voronoi_instance.finalise_DCEL()
    utils.clear_canvas(ctx)
    draw_sites(ctx, voronoi_instance)
    draw_beach_line_components(ctx, voronoi_instance)
    draw_sweep_line(ctx, voronoi_instance)
    draw_circle_events(ctx, voronoi_instance)
    utils.drawDCEL(ctx, dcel)
        
def draw_sites(ctx, voronoi_instance):
    ctx.set_source_rgba(*SITE_COLOUR)
    for site in voronoi_instance.sites:
        utils.drawCircle(ctx, *site.loc, SITE_RADIUS)

def draw_circle_events(ctx, voronoi_instance):
    for event in voronoi_instance.circles:
        if event.active:
            ctx.set_source_rgba(*CIRCLE_COLOUR)
            utils.drawCircle(ctx, *event.loc, CIRCLE_RADIUS)
        else:
            ctx.set_source_rgba(*CIRCLE_COLOUR_INACTIVE)
            utils.drawCircle(ctx, *event.loc, CIRCLE_RADIUS)
                
def draw_beach_line_components(ctx, voronoi_instance):
    #the arcs themselves
    ctx.set_source_rgba(*BEACH_LINE_COLOUR, 0.1)
    xs = np.linspace(0,1,2000)
    for arc in voronoi_instance.beachline.arcs_added:
        xys = arc(xs)
        for x,y in xys:
            utils.drawCircle(ctx, x, y, BEACH_RADIUS)
    #--------------------
    #the frontier:
    # Essentially a horizontal travelling sweep line to draw segments
    ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
    leftmost_x = nan
    ##Get the chain of arcs:
    chain = voronoi_instance.beachline.get_chain()
    if len(chain) > 1:
        enumerated = list(enumerate(chain))
        pairs = zip(enumerated[0:-1],enumerated[1:])
        for (i,a),(j,b) in pairs:
            logging.debug("Drawing {} -> {}".format(a,b))
            intersections = a.value.intersect(b.value, voronoi_instance.sweep_position.y())
            logging.debug("Intersections: ",intersections)
            if len(intersections) == 0:
                logging.exception("NO INTERSECTION: {} - {}".format(i,j))
                #Draw the non-intersecting line as red
                ctx.set_source_rgba(*BEACH_NO_INTERSECT_COLOUR)
                xs = np.linspace(0, 1.0, 2000)
                axys = a.value(xs)
                bxys = b.value(xs)
                for x,y in axys:
                    utils.drawCircle(ctx,x,y,BEACH_RADIUS)
                for x,y in bxys:
                    utils.drawCircle(ctx,x,y,BEACH_RADIUS)
                ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
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
                utils.drawCircle(ctx, x, y, BEACH_RADIUS)

    if len(chain) > 0 and (leftmost_x is nan or leftmost_x < 1.0):
        if leftmost_x is nan:
            leftmost_x = 0
        #draw the last arc:
        logging.debug("Final Arc from {0:.2f} to {1:.2f}".format(leftmost_x,1.0))
        xs = np.linspace(leftmost_x,1.0,2000)
        frontier_arc = chain[-1].value(xs)
        for x,y in frontier_arc:
            utils.drawCircle(ctx,x,y,BEACH_RADIUS)
            
def draw_sweep_line(ctx, voronoi_instance):
    if voronoi_instance.sweep_position is None:
        return        
    ctx.set_source_rgba(*SWEEP_LINE_COLOUR)
    ctx.set_line_width(LINE_WIDTH)
    #a tuple
    sweep_event = voronoi_instance.sweep_position
    ctx.move_to(0.0,sweep_event.y())
    ctx.line_to(1.0,sweep_event.y())
    ctx.close_path()
    ctx.stroke()
