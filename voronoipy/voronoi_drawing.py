    #-------------------- DEBUG Drawing Methods
    def draw_voronoi_diagram(self,clear=True):
        """ Draw the final diagram """
        logging.debug("Drawing final voronoi diagram")
        if clear:
            utils.clear_canvas(self.ctx)
        self.ctx.set_source_rgba(*COLOUR)
        #draw sites
        for site in self.sites:
            utils.drawCircle(self.ctx,*site.loc,0.007)
        #draw faces
        utils.drawDCEL(self.ctx,self.dcel)                       

    def draw_intermediate_states(self):
        logging.debug("Drawing intermediate state")
        utils.clear_canvas(self.ctx)
        self.draw_sites()
        self.draw_beach_line_components()
        self.draw_sweep_line()
        self.draw_circle_events()
        utils.drawDCEL(self.ctx,self.dcel)
        
    def draw_sites(self):
        self.ctx.set_source_rgba(*SITE_COLOUR)
        for site in self.sites:
            utils.drawCircle(self.ctx,*site.loc,SITE_RADIUS)

    def draw_circle_events(self):
        for event in self.circles:
            if event.active:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR)
                utils.drawCircle(self.ctx,*event.loc,CIRCLE_RADIUS)
            else:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR_INACTIVE)
                utils.drawCircle(self.ctx,*event.loc,CIRCLE_RADIUS)
                
    def draw_beach_line_components(self):
        #the arcs themselves
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR,0.1)
        xs = np.linspace(0,1,2000)
        for arc in self.beachline.arcs_added:
            xys = arc(xs)
            for x,y in xys:
                utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
        #--------------------
        #the frontier:
        # Essentially a horizontal travelling sweep line to draw segments
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
        leftmost_x = nan
        ##Get the chain of arcs:
        chain = self.beachline.get_chain()
        if len(chain) > 1:
            enumerated = list(enumerate(chain))
            pairs = zip(enumerated[0:-1],enumerated[1:])
            for (i,a),(j,b) in pairs:
                logging.debug("Drawing {} -> {}".format(a,b))
                intersections = a.value.intersect(b.value,self.sweep_position.y())
                #print("Intersections: ",intersections)
                if len(intersections) == 0:
                    logging.exception("NO INTERSECTION: {} - {}".format(i,j))
                    #Draw the non-intersecting line as red
                    self.ctx.set_source_rgba(*BEACH_NO_INTERSECT_COLOUR)
                    xs = np.linspace(0,1,2000)
                    axys = a.value(xs)
                    bxys = b.value(xs)
                    for x,y in axys:
                        utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
                    for x,y in bxys:
                        utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)
                    self.ctx.set_source_rgba(*BEACH_LINE_COLOUR2,1)
                    continue
                    #----------
                    #raise Exception("No intersection point")
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
                xs = np.linspace(leftmost_x,left_most_intersection,2000)
                #update the position
                leftmost_x = left_most_intersection
                frontier_arc = a.value(xs)
                for x,y in frontier_arc:
                    utils.drawCircle(self.ctx,x,y,BEACH_RADIUS)

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
        if self.sweep_position is None:
            return        
        self.ctx.set_source_rgba(*SWEEP_LINE_COLOUR)
        self.ctx.set_line_width(LINE_WIDTH)
        #a tuple
        sweep_event = self.sweep_position
        self.ctx.move_to(0.0,sweep_event.y())
        self.ctx.line_to(1.0,sweep_event.y())
        self.ctx.close_path()
        self.ctx.stroke()


