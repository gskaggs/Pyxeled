
# Configuration

T = 200
T_final = 1
alpha = 0.7
delta = 1.0 


#######################################################################################################

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
        color_diff = (sum([(in_color[i] - pallete_color[i])**2 for i in range(3)]))**0.5
        
        spatial_diff = ((x-x0)**2 + (y-y0)**2)**0.5

        return color_diff + 40 * ((N / M)**0.5) * spatial_diff;


    def add_pixel(self, x0, y0):
        pixels.add((x0, y0))

    def clear_pixels(self):
        pixels = set()

#def update_pos():

    # update pallete color as well
    def norm_probs(self):
        global colors
        denom = sum(p_c)
        hi = max(p_c)
        
        for i in range(len(p_c)):
            if p_c[i] == hi:
                pallete_color = colors[i].color
            p_c[i] /= denom


    def update_sp_color(self, c):
        sp_color = c


class Color:
    
    def __init__(self, c, p):
        self.color, self.probability = c, p

    def condit_prob(self, sp):

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

M = w_in * h_in 
N = h_in * h_out

X = [(r * w_in) // w_out for r in range(w_out)]
Y = [(c * h_in) // h_out for c in range(h_out)]
init_color = avg_color(in_image, M)
super_pixels = [[SuperPixel(x,y,init_color) for x in X] for y in Y]

clusters = [(0,1)]
colors = [Color(init_color, 0.5), Color(init_color, 0.5)]
colors[1].perturb()

