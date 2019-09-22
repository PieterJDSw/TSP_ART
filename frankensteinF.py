import urllib.request  
import os  

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
plt.style.use('seaborn-pastel')
  
image_url = 'http://ereaderbackgrounds.com/movies/bw/Frankenstein.jpg'  
image_path = 'madib.jpg'  
  
if not os.path.exists(image_path):  
    urllib.request.urlretrieve(image_url, image_path)
	
from PIL import Image  
  
original_image = Image.open(image_path)  
bw_image = original_image.convert('1', dither=Image.NONE)  
# bw_image.show()


 ############################################################
bw_image_array = np.array(bw_image, dtype=np.int)  
black_indices = np.argwhere(bw_image_array == 0)  
# Changing "size" to a larger value makes this algorithm take longer,  
# but provides more granularity to the portrait  
chosen_black_indices = black_indices[  
                           np.random.choice(black_indices.shape[0],  
                                            replace=False,  
                                            size=1000)]  
  
print(chosen_black_indices[0]) 
# plt.scatter([x[1] for x in chosen_black_indices],  
#             [x[0] for x in chosen_black_indices],  
#             color='black', s=1)  
# plt.gca().invert_yaxis()  

# plt.xticks([])  
# plt.yticks([])
# plt.show()
# print(chosen_black_indices)



###################################
from scipy.spatial.distance import pdist, squareform  
  
distances = pdist(chosen_black_indices)  
distance_matrix = squareform(distances)
##########################################
from tsp_solver.greedy_numpy import solve_tsp  
  
optimized_path = solve_tsp(distance_matrix)  
  
optimized_path_points = [chosen_black_indices[x] for x in optimized_path]
print(optimized_path_points[0])

plt.figure(figsize=(8, 10), dpi=100)  
plt.plot([x[1] for x in optimized_path_points],  
         [x[0] for x in optimized_path_points],  
         color='black', lw=1)  
plt.xlim(0, 600)  
plt.ylim(0, 800)  
# plt.gca().invert_yaxis()  
plt.xticks([])  
plt.yticks([])
plt.show()
############################
fig = plt.figure(figsize=(16, 10), dpi=100)  
  
plt.subplot(1, 2, 1)  
plt.imshow(original_image)  
plt.grid(False)  
plt.xlim(0, 600)  
plt.ylim(0, 800)  
plt.gca().invert_yaxis()  
plt.xticks([])  
plt.yticks([])  
  
plt.subplot(1, 2, 2)  

plt.plot([x[1] for x in optimized_path_points],  
         [x[0] for x in optimized_path_points],  
         color='black', lw=1)  
plt.grid(False)  
plt.xlim(0, 600)  
plt.ylim(0, 800)  
plt.gca().invert_yaxis()  
plt.xticks([])  
plt.yticks([])
##################################### animation start
counter = 0
xstartdata = optimized_path_points[0][0]
ystartdata = optimized_path_points[0][1]
fig, ax = plt.subplots()
ln, = plt.plot([], [], '--bo', animated=True)
plt.plot([xstartdata], [ystartdata], '--bo',  lw=5)
plt.gca().invert_yaxis()
x = []
y = []

def init():
    ln.set_data([],[])
    # ax.set_xlim(0, 1000)
    # ax.set_ylim(0, 1000)
    return ln,

def animate(i):
    x.append(optimized_path_points[i][1])
    y.append(optimized_path_points[i][0])
    
    ln.set_data(x, y)
    plt.plot(x , y , lw=1,animated=True)
    return ln,

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=range(len(optimized_path_points)), interval=5, blit=True)
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
#sys.exit()
