
# Configuration

T = 200
T_final = 1
alpha = 0.7
delta = 1.0 
e = 2.71828

K = 1
K_max = 8

M = w_in * h_in 
N = h_in * h_out


#######################################################################################################
def color_diff(c1, c2):
    res = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2
    res = res**0.5
    return res #(sum( [(c1[i] - c2[i])**2 for i in range(3)] ))**0.5

class SuperPixel:
    pixels = set()
    p_c = [1]
    sp_color = (0, 0 , 0)

    def __init__(self, x, y, c):
        global N
        self.x, self.y, self.pallete_color = x, y, c
        self.p_s = 1 / N

    def cost(self, x0, y0):
        global in_image

        in_color = in_image[x0][y0]
        c_diff = color_diff(in_color, pallete_color)         
        spatial_diff = ((x-x0)**2 + (y-y0)**2)**0.5

        return c_diff + 45 * ((N / M)**0.5) * spatial_diff;


    def add_pixel(self, x0, y0):
        pixels.add((x0, y0))

    def clear_pixels(self):
        pixels = set()

    def update_pos():
        x, y = 0, 0
        for pxl in pixels:
            x += pxl[0]
            y += pxl[1]
        x /= len(pixels)
        y /= len(pixels)

    # update pallete color as well
    def normalize_probs(self):
        global palette 
        denom = sum(p_c)
        hi = max(p_c)
        
        for i in range(len(p_c)):
            if p_c[i] == hi:
                pallete_color = palette[i].color
            p_c[i] /= denom


    def update_sp_color(self):
        global in_image
        c = [0, 0, 0]        
        
        for pxl in pixels:
            c += in_image[pxl[0]][pxl[1]]
        
        for i in range(3):
            c[i] /= len(pixels)

        sp_color = tuple(c)



class Color:
    
    def __init__(self, c, p):
        self.color, self.probability = c, p

    def condit_prob(self, sp):
        global T, e
        return probability * (-1 * e ** (color_diff(sp.sp_color, color) / T))

    def perturb(self):
        global delta
        color = (color[0] + delta, color[1], color[2])

#######################################################################################################

# Initialize super pixels and color pallete

def avg_color(in_image, M):
    res = [0, 0, 0]
    for row in in_image:
        for c in row:
            for i in range(3):
                res[i] += c[i] 

    for i in range(3):
        res[i] /= M

    return tuple(res)


X = [(r * w_in) // w_out for r in range(w_out)]
Y = [(c * h_in) // h_out for c in range(h_out)]
init_color = avg_color(in_image, M)
super_pixels = [[SuperPixel(x,y,init_color) for x in X] for y in Y]

clusters = [(0,1)]
palette = [Color(init_color, 0.5), Color(init_color, 0.5)]
palette[1].perturb()

#######################################################################################################

def sp_refine():
    global super_pixels, in_image
    for row in super_pixels:
        for sp in row:
            sp.clear_pixels()

    # Update pixel association
    for x in range(w_in):
        for y in range(h_in):
            best_pair = (-1, -1)
            best_cost = 10**9

            for r in range(w_out):
                for c in range(h_out):
                    cur = sp[r][c].cost(x, y)
                    if cur < best_cost:
                        best_cost = cur
                        best_pair = (r, c)

            sp[best_pair[0]][best_pair[1]].add_pixel(x, y)

    # Update color and position
    for row in super_pixels:
        for sp in row:
            sp.update_pos()
            sp.update_color()



def associate():
    global super_pixels, palette 
    for sp in super_pixels:
        sp.p_c = [0] * 2 * K
        for k in range(2 * K):
            sp.p_c[k] = palette[k].condit_prob(sp)
        sp.normalize_probs()
            

    for k in range(2 * K):
        palette[k].probability = 0
        for sp in super_pixels:
            palette[k].probability += sp.p_c[k] * sp.p_s

def palette_refine():
    global super_pixels, palette 
    for k in range(2 * K):
        new_color = [0, 0, 0]
        for sp in super_pixels:
            for i in range(3):
                new_color[i] += (sp.sp_color[i] * sp.p_c[k] * sp.p_s) / palette[k].probability

        palette[k].color = tuple(new_color)


    
