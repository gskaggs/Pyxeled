
# Configuration

from skimage import color as color_lib
from PIL import Image

image = Image.open("input.png")
image_data = image.load()

w_in, h_in = image.size 
in_rgb = [[list(image_data[r, c]) for c in range(h_in)] for r in range(w_in)]
in_rgb = [[[in_rgb[r][c][i] / 255 for i in range(3)] for c in range(h_in)] for r in range(w_in)]

in_image = color_lib.rgb2lab(in_rgb)

T = 25 
T_final = 3 
alpha = 0.7
delta = 1.5 
e = 2.71828
epsilon_palette = 1
epsilon_cluster = 0.25

K = 1
K_max = 8

w_out = 22
h_out = 32

M = w_in * h_in 
N = w_out * h_out



#######################################################################################################
def color_diff(c1, c2):
    res = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2
    res = res**0.5
    return res #(sum( [(c1[i] - c2[i])**2 for i in range(3)] ))**0.5

class SuperPixel:
    def __init__(self, x, y, c):
        global N
        self.x, self.y, self.pallete_color = x, y, c
        self.p_s = 1 / N
        self.pixels = set()
        self.p_c = [0.5, 0.5]
        self.sp_color = (0, 0 , 0)

    def cost(self, x0, y0):
        global in_image

        in_color = in_image[x0][y0]
        c_diff = color_diff(in_color, self.pallete_color)         
        spatial_diff = ((self.x-x0)**2 + (self.y-y0)**2)**0.5

        return c_diff + 45 * ((N / M)**0.5) * spatial_diff;


    def add_pixel(self, x0, y0):
        self.pixels.add((x0, y0))

    def clear_pixels(self):
        self.pixels = set()

    def update_pos(self):
        self.x, self.y = 0, 0
        for pxl in self.pixels:
            self.x += pxl[0]
            self.y += pxl[1]
        self.x /= len(self.pixels)
        self.y /= len(self.pixels)

    # update pallete color as well
    def normalize_probs(self):
        global palette, clusters, K
        denom = sum(self.p_c)
        hi = max(self.p_c)
        
        for i in range(len(self.p_c)):
            if self.p_c[i] == hi:
                self.pallete_color = palette[i].color
            self.p_c[i] /= denom

        hi = -1
        for k in range(K):
            cluster = clusters[k]
            prob = 0
            color = [0, 0, 0]
            for i in range(len(cluster)):
                cur = palette[cluster[i]]
                prob += cur.probability
                for j in range(3):
                    color[j] += cur.color[j]
            for j in range(3):
                color[j] /= len(cluster)
            if prob > hi:
                hi = prob
                self.palette_color = color



    def update_sp_color(self):
        global in_image
        c = [0, 0, 0]        
        
        for pxl in self.pixels:
            for i in range(3):
                c[i] += in_image[pxl[0]][pxl[1]][i]
        
        for i in range(3):
            c[i] /= len(self.pixels)

        self.sp_color = tuple(c)
#print(self.sp_color[0], self.sp_color[0], self.sp_color[0])



class Color:

    def __init__(self, c, p):
        self.color, self.probability = c, p

    def condit_prob(self, sp):
        global T, e
        return self.probability * (e ** (-1 * color_diff(sp.sp_color, self.color) / T))

    def perturb(self):
        global delta
        self.color = (self.color[0] + delta, self.color[1] + delta, self.color[2] + delta)

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
super_pixels = [[SuperPixel(x,y,init_color) for y in Y] for x in X]


clusters = [(0,1)]
palette = [Color(init_color, 0.5), Color(init_color, 0.5)]
palette[1].perturb()

#######################################################################################################
def in_bounds(r, c):
    global w_out, h_out
    return r >= 0 and c >= 0 and r < w_out and c < h_out

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
            
            dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            dy = [-1, 0, 1, -1, 0, 1,-1, 0, 1] 
            r = (x * w_out) // w_in
            c = (y * h_out) // h_in

            for i in range(9):
                if in_bounds(dx[i]+r, dy[i]+c):
                    cur = super_pixels[dx[i]+r][dy[i]+c].cost(x, y)
                    if cur < best_cost:
                        best_cost = cur
                        best_pair = (dx[i]+r, dy[i]+c)

            super_pixels[best_pair[0]][best_pair[1]].add_pixel(x, y)

    # Update color and position
    for row in super_pixels:
        for sp in row:
            sp.update_pos()
            sp.update_sp_color()

def associate():
    global super_pixels, palette 
    for row in super_pixels:
        for sp in row:
            sp.p_c = [0] * (len(palette))
            for k in range(len(palette)):
                sp.p_c[k] = palette[k].condit_prob(sp)
            sp.normalize_probs()
# for p in sp.p_c:
#                print(p, end=" ")
#            print()
            

    for k in range(len(palette)):
        palette[k].probability = 0

        for row in super_pixels:
            for sp in row:
                palette[k].probability += sp.p_c[k] * sp.p_s
        print("P_", k, palette[k].probability)

def palette_refine():
    global super_pixels, palette 
    total_change = 0
    for k in range(len(palette)):
        new_color = [0, 0, 0]
        for row in super_pixels:
            for sp in row:
                for i in range(3):
                    new_color[i] += (sp.sp_color[i] * sp.p_c[k] * sp.p_s) / palette[k].probability

        old_color = palette[k].color
        palette[k].color = tuple(new_color)
        total_change += color_diff(old_color, new_color)

    return total_change

def expand():
    global clusters, palette, epsilon_cluster, K, K_max

    for i in range(K):
        if (K >= K_max):
            break


        c1 = palette[clusters[i][0]]
        c2 = palette[clusters[i][1]]

        if color_diff(c1.color, c2.color) > epsilon_cluster:
            K += 1
            palette.append(Color(c1.color, c1.probability / 2))
            palette.append(Color(c2.color, c2.probability / 2))
            c1.probability /= 2
            c2.probability /= 2


            clusters.append((clusters[i][1], len(palette)-1))
            clusters[i] = (clusters[i][0], len(palette)-2)

            assert abs(palette[clusters[i][0]].probability - palette[clusters[i][1]].probability) < epsilon_cluster
            assert abs(palette[clusters[-1][0]].probability - palette[clusters[-1][1]].probability) < epsilon_cluster

    if K >= K_max:
        new_palette = []
        new_clusters = [] 
        for k in range(K):
            c = clusters[k]
            if len(c) == 2:
                c1 = palette[c[0]]
                c2 = palette[c[1]]
                new_color = [0,0,0]
                for i in range(3):
                    new_color[i] = (c1.color[i] + c2.color[i])/2
                cur = Color(tuple(new_color), c1.probability + c2.probability)
                new_palette.append(cur)
                new_clusters.append(tuple([k]))
            else:
                assert False

        palette = new_palette
        clusters = new_clusters
                

    else:
        # So sub-clusters can separate
        for i in range(K):
            c = palette[clusters[i][1]]
            c.perturb()

def saturate(out_lab):
    for r in range(w_out):
        for c in range(h_out):
            out_lab[r][c][1] *= 1.1
            out_lab[r][c][2] *= 1.1


#######################################################################################################

iterations = 0
while T > T_final:
#for i in range(3):
    print("K", K)
    print("T", T)
    print("iterations", iterations)

    for k in range(K):
        for h in range(len(clusters[k])):
            for j in range(3):
                print(palette[clusters[k][h]].color[j], end=" ")
            print()
        print()

    iterations += 1

    sp_refine()

    associate()

    total_change = palette_refine()

    if total_change < epsilon_palette:
        T *= alpha
        if K < K_max:
            expand()

    print()



out_lab = []
for r in range(w_out):
    cur = []
    for c in range(h_out):
        # Should use cluster, not sub-cluster  pallete_color
        cur.append(list(super_pixels[r][c].pallete_color))
    out_lab.append(cur)

saturate(out_lab)

out_image = color_lib.lab2rgb(out_lab)
out_image = [[[int(round(out_image[r][c][i] * 255)) for i in range(3)] for c in range(h_out)] for r in range(w_out)] 


output = Image.new("RGB", (w_out, h_out)) 
out_data = output.load()
for r in range(w_out):
    for c in range(h_out):
        out_data[r,c] = tuple(out_image[r][c])

output.save("output.png")
exit()

    


