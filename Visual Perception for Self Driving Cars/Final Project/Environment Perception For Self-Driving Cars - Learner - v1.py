#!/usr/bin/env python
# coding: utf-8

# # Environment Perception For Self-Driving Cars
# 
# Welcome to the final assignment for this course. In this assignment, you will learn how to use the material so far to extract useful scene information to allow self-driving cars to safely and reliably traverse their environment.
# 
# 
# **In this assignment, you will:**
# - Use the output of semantic segmentation neural networks to implement drivable space estimation in 3D.
# - Use the output of semantic segmentation neural networks to implement lane estimation.
# - Use the output of semantic segmentation to filter errors in the output of 2D object detectors. 
# - Use the filtered 2D object detection results to determine how far obstacles are from the self-driving car.
# 
# For most exercises, you are provided with a suggested outline. You are encouraged to diverge from the outline if you think there is a better, more efficient way to solve a problem.
# 
# Please go through cells in order. Lower cells will depend on higher cells to work properly.
# 
# You are only allowed to use the packages loaded bellow, mainly numpy, OpenCV, and the custom functions explained in the notebook. Run the cell bellow to import the required packages:

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from m6bk import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)
np.set_printoptions(precision=2, threshold=np.nan)


# ## 0 - Loading and Visualizing the Data
# We provide you with a convenient dataset handler class to read and iterate through samples taken from the CARLA simulator. Run the following code to create a dataset handler object. 

# In[2]:


dataset_handler = DatasetHandler()


# The dataset handler contains three test data frames 0, 1, and 2. Each frames contains:
# - DatasetHandler().rgb: a camera RGB image 
# - DatasetHandler().depth: a depth image containing the depth in meters for every pixel.
# - DatasetHandler().segmentation: an image containing the output of a semantic segmentation neural network as the category per pixel. 
# - DatasetHandler().object_detection: a numpy array containing the output of an object detection network. 
# 
# Frame 0 will be used throughout this note book as a running example. **Frame 1 will be used for submission and grading of this assesment.** Frame 2 is provided as a challange for learners interested in a more difficult example.
# 
# The current data frame being read can be known through the following line of code:

# In[3]:


dataset_handler.current_frame


#  Upon creation of the dataset handler object, frame 0 will be automatically read and loaded. The frame contents can be accessed by using four different attributes of the dataset handler object: image, depth, object_detection, and semantic segmentation. As an example, to access the image, camera calibration matrix, and depth run the following three cells:    

# In[4]:


image = dataset_handler.image
plt.imshow(image)


# In[5]:


k = dataset_handler.k
print(k)


# In[6]:


depth = dataset_handler.depth
plt.imshow(depth, cmap='jet')


# The semantic segmentation output can be accessed in a similar manner through:

# In[7]:


segmentation = dataset_handler.segmentation
plt.imshow(segmentation)


# ### Segmentation Category Mappings:
# The output segmentation image contains mapping indices from every pixel to a road scene category. To visualize the semantic segmentation output, we map the mapping indices to different colors. The mapping indices and visualization colors for every road scene category can be found in the following table: 
# 
# |Category |Mapping Index| Visualization Color|
# | --- | --- | --- |
# | Background | 0 | Black |
# | Buildings | 1 | Red |
# | Pedestrians | 4 | Teal |
# | Poles | 5 | White |
# | Lane Markings | 6| Purple |
# | Roads | 7 | Blue |
# | Side Walks| 8 | Yellow |
# | Vehicles| 10 | Green |
# 
# The vis_segmentation function of the dataset handler transforms the index image to a color image for visualization: 

# In[8]:


colored_segmentation = dataset_handler.vis_segmentation(segmentation)

plt.imshow(colored_segmentation)


# The set_frame function takes as an input a frame number from 0 to 2 and loads that frame allong with all its associated data. It will be useful for testing and submission at the end of this assesment.

# In[9]:


dataset_handler.set_frame(2)
dataset_handler.current_frame


# In[10]:


image = dataset_handler.image
plt.imshow(image)


# ## 1 - Drivable Space Estimation Using Semantic Segmentation Output
# 
# Your first task is to implement drivable space estimation in 3D. You are given the output of a semantic segmentation neural network, the camera calibration matrix K, as well as the depth per pixel.
# 
# ### 1.1 - Estimating the x, y, and z coordinates of every pixel in the image:
# 
# You will be using the equations learned in module 1 to compute the x, y, and z coordinates of every pixel in the image. As a reminder, the equations to get the required 3D coordinates are:
# 
# $$z = depth $$
# 
# $x = \frac{(u - c_u) * z}{f} \tag{1}$
# 
# $y = \frac{(v - c_v) * z}{f} \tag{2}$
# 
# Here, $c_u$, $c_v$, and $f$ are the intrinsic calibration parameters found in the camera calibration matrix K such that:
# 
# $$K = \begin{pmatrix} f & 0 & u_c \\ 0 & f & u_v \\ 0& 0 & 1 \end{pmatrix}$$
# 
# ***Note***: Make sure you are on frame 0 for the rest of this assessment. You will use the rest of the frames for testing after the assessment is done.
# 
# **Exercise**: Implement the estimation of the x and y coordinates of every pixel using equations (1) and (2):

# In[11]:


# GRADED FUNCTION: xy_from_depth
def xy_from_depth(depth, k):
    """
    Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.

    Arguments:
    depth -- tensor of dimension (H, W), contains a depth value (in meters) for every pixel in the image.
    k -- tensor of dimension (3x3), the intrinsic camera matrix

    Returns:
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    """
    ### START CODE HERE ### (≈ 7 lines in total)

    # Get the shape of the depth tensor
    h,w = depth.shape

    # Grab required parameters from the K matrix
    f = k[0][0]
    c_u = k[0][2]
    c_v = k[1][2]

    # Generate a grid of coordinates corresponding to the shape of the depth map
    n_x,n_y = (h,w)
    xa = np.arange(1, w+1, 1)
    ya = np.arange(1, h+1, 1)
    u,v = np.meshgrid(xa, ya)
    # Compute x and y coordinates
    x = ((u-c_u)*depth)/f
    y = ((v-c_v)*depth)/f

    ### END CODE HERE ###
    
    return x, y


# In[12]:


dataset_handler.set_frame(0)

k = dataset_handler.k

z = dataset_handler.depth

x, y = xy_from_depth(z, k)

print('x[800,800] = ' + str(x[800, 800]))
print('y[800,800] = ' + str(y[800, 800]))
print('z[800,800] = ' + str(z[800, 800]) + '\n')

print('x[500,500] = ' + str(x[500, 500]))
print('y[500,500] = ' + str(y[500, 500]))
print('z[500,500] = ' + str(z[500, 500]) + '\n')


# **Expected Output**:
# 
# $x[800,800] = 0.720$
# <br />
# $y[800,800] = 1.436$
# <br />
# $z[800,800] = 2.864$
# 
# $x[500,500] = -9.5742765625$
# <br />
# $y[500,500] = 1.4464734375$
# <br />
# $z[500,500] = 44.083$

# ### 1.2 - Estimating The Ground Plane Using RANSAC:
# 
# In the context of self-driving cars, drivable space includes any space that the car is physically capable of traversing in 3D. The task of estimating the drivable space is equivalent to estimating pixels belonging to the ground plane in the scene. For the next exercise, you will use RANSAC to estimate the ground plane in the 3D camera coordinate frame from the x,y, and z coordinates estimated above. 
# 
# The first step is to process the semantic segmentation output to extract the relevant pixels belonging to the class you want consider as ground. For this assessment, that class is the road class with a mapping index of 7. To extract the x,y,z coordinates of the road, run the following cell:

# In[13]:


# Get road mask by choosing pixels in segmentation output with value 7
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1

# Show road mask
plt.imshow(road_mask)

# Get x,y, and z coordinates of pixels in road mask
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))


# The next step is to use the extracted x, y, and z coordinates of pixels belonging to the road to estimate the ground plane. RANSAC will be used for robustness against outliers.

# **Exercise**: Implement RANSAC for plane estimation. Here are the 6 steps:
# 1. Choose a minimum of 3 points from xyz_ground at random.
# 2. Compute the ground plane model using the chosen random points, and the provided function compute_plane.
# 3. Compute the distance from the ground plane model to every point in xyz_ground, and compute the number of inliers based on a distance threshold.
# 4. Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.  
# 5. Repeat until number of iterations $\geq$ a preset number of iterations, or number of inliers $\geq$ minimum number of inliers.
# 6. Recompute and return a plane model using all inliers in the final inlier set. 
# 
# Useful functions: `np.random.choice()`, `compute_plane()`, `dist_to_plane()`.
# 
# 
# The two custom functions are provided to help you finish this part:
# 
# 1. ***compute_plane(xyz):***
# ```
#     Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
# 
#     Arguments:
#     xyz -- tensor of dimension (3, N), contains points needed to fit plane.
#     k -- tensor of dimension (3x3), the intrinsic camera matrix
# 
#     Returns:
#     p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
# ```
# 
# 2. ***dist_to_plane(plane, x, y, z):***
# 
# ```
#     Computes distance from N points to a plane in 3D, given the plane parameters and the x,y,z coordinate of points.
# 
#     Arguments:
#     plane -- tensor of dimension (4,1), containing the plane parameters [a,b,c,d]
#     x -- tensor of dimension (Nx1), containing the x coordinates of the points
#     y -- tensor of dimension (Nx1), containing the y coordinates of the points
#     z -- tensor of dimension (Nx1), containing the z coordinates of the points
# 
# 
#     Returns:
#     distance -- tensor of dimension (N, 1) containing the distance between points and the plane
# ```
# 
# The functions are already loaded through the import statement at the beginning of the notebook. You also could perform plane estimation yourself if you are up for a challenge!

# In[22]:


# GRADED FUNCTION: RANSAC Plane Fitting

def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr -- 
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """
    
    ### START CODE HERE ### (≈ 23 lines in total)
    
    # Set thresholds:
    num_itr = 10000  # RANSAC maximum number of iterations
    min_num_inliers = 30000  # RANSAC minimum number of inliers
    distance_threshold = 1e-6  # Maximum distance from point to plane for point to be considered inlier
    #tried a lot of values ranging from +- 3 times the values, but didn't affect much
    N = xyz_data.shape[1]
    c_temp = np.zeros((3,4))
    i_temp = []
    for i in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        ac = np.random.choice(N,5)
        c = (np.vstack((xyz_data.T[ac[0]], xyz_data.T[ac[1]], xyz_data.T[ac[2]], xyz_data.T[ac[3]], xyz_data.T[ac[4]]))).T
        # Step 2: Compute plane model
        p = compute_plane(c)
        # Step 3: Find number of inliers
        b = np.random.choice(N,150)
        x = np.array(xyz_data[0,0:]).T
        y = np.array(xyz_data[1,0:]).T
        z = np.array(xyz_data[2,0:]).T
        
        d = dist_to_plane(p,x,y,z)
        # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        inline=np.zeros(d.shape)
        inline[d<distance_threshold] = 1
        if sum(inline) > sum(i_temp):
            c_temp = c
        
        # Step 5: Check if stopping criterion is satisfied and break.         
        if sum(inline)>= min_num_inliers:
            return p
        
    # Step 6: Recompute the model parameters using largest inlier set.         
    p = compute_plane(c_temp)
    ### END CODE HERE ###
    
    return p


# In[23]:


p_final = ransac_plane_fit(xyz_ground)
print('Ground Plane: ' + str(p_final))


# **Expected Output**:
# 
# Ground Plane: [0.01791606 -0.99981332  0.00723433  1.40281479]

# To verify that the estimated plane is correct, we can visualize the inlier set computed on the whole image. Use the cell bellow to compute and visualize the ground mask in 2D image space.

# In[24]:


dist = np.abs(dist_to_plane(p_final, x, y, z))

ground_mask = np.zeros(dist.shape)

ground_mask[dist < 0.1] = 1
ground_mask[dist > 0.1] = 0

plt.imshow(ground_mask)


# We also provide a function to visualize the estimated drivable space in 3D. Run the following cell to visualize your estimated drivable space in 3D.

# In[25]:


dataset_handler.plot_free_space(ground_mask)


# The above visualization only shows where the self-driving car can physically travel. The obstacles such as the SUV to the left of the image, can be seen as dark pixels in our visualization: 

# <tr>
# <td> <img src="images/image.png" style="width:320px;height:240px;">   </td>
# <td> <img src="images/occ_grid.png" style="width:240px;height:240px;">   </td>
# </tr>

# However, estimating the drivable space is not enough for the self-driving car to get on roads. The self-driving car still needs to perform lane estimation to know where it is legally allowed to drive. Once you are comfortable with the estimated drivable space, continue the assessment to estimate the lane where the car can drive.
# 
# ## 2 - Lane Estimation Using The Semantic Segmentation Output
# 
# Your second task for this assessment is to use the output of semantic segmentation to estimate the lane boundaries of the current lane the self-driving car is using. This task can be separated to two subtasks, lane line estimation, and post-processing through horizontal line filtering and similar line merging.
# 
# ### 2.1 Estimating Lane Boundary Proposals:
# The first step to perform this task is to estimate any line that qualifies as a lane boundary using the output from semantic segmentation. We call these lines 'proposals'.
# 
# **Exercise**: Estimate lane line proposals using OpenCv functions. Here are the 3 steps:
# 1. Create an image containing the semantic segmentation pixels belonging to categories relevant to the lane boundaries, similar to what we have done previously for the road plane. For this assessment, these pixels have the value of 6 and 8 in the neural network segmentation output.
# 2. Perform edge detection on the derived lane boundary image.
# 3. Perform line estimation on the output of edge detection.
# 
# Useful functions: `cv2.Canny()`, `cv2.HoughLinesP()`, `np.squeeze()`.

# In[65]:


# GRADED FUNCTION: estimate_lane_lines
def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """
    ### START CODE HERE ### (≈ 7 lines in total)
    # Step 1: Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    img = np.zeros(segmentation.shape)
    img[segmentation_output==6] = 1 #road markings
    img[segmentation_output==8] = 1 #sidewalks

    # Step 2: Perform Edge Detection using cv2.Canny()
    edges = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

    # Step 3: Perform Line estimation using cv2.HoughLinesP()
    edges = np.uint8(edges)
    lines = cv2.HoughLinesP(edges, rho = 1, theta = np.pi/180, threshold = 400, minLineLength = 100, maxLineGap = 10)
    lines = lines.reshape((lines.shape[0],lines.shape[2]))
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    ### END CODE HERE ###

    return lines


# In[66]:


lane_lines = estimate_lane_lines(segmentation)

print(lane_lines.shape)

plt.imshow(dataset_handler.vis_lanes(lane_lines))


# ***Expected Output***
# 
# <img src="images/lanes_1.png" style="width:320px;height:240px;"> 

# ### 2.2 - Merging and Filtering Lane Lines:
# 
# The second subtask to perform the estimation of the current lane boundary is to merge redundant lines, and filter out any horizontal lines apparent in the image. Merging redundant lines can be solved through grouping lines with similar slope and intercept. Horizontal lines can be filtered out through slope thresholding. 
# 
# **Exercise**: Post-process the output of the function ``estimate_lane_lines`` to merge similar lines, and filter out horizontal lines using the slope and the intercept. The three steps are:
# 1. Get every line's slope and intercept using the function provided.
# 2. Determine lines with slope less than horizontal slope threshold. Filtering can be performed later if needed.
# 3. Cluster lines based on slope and intercept as you learned in Module 6 of the course. 
# 4. Merge all lines in clusters using mean averaging.
# 
# Usefull Functions:
# 1. ***get_slope_intecept(lines):***
# 
# ```
#     Computes distance from N points to a plane in 3D, given the plane parameters and the x,y,z coordinate of points.
# 
#     Arguments:
#     lines -- tensor of dimension (N,4) containing lines in the form of [x_1, y_1, x_2, y_2], the coordinates of two points on       the line
# 
#     Returns:
#     slopes -- tensor of dimension (N, 1) containing the slopes of the lines
#     intercepts -- tensor of dimension (N,1) containing the intercepts of the lines
# ```
# 
# This function is already loaded through the import statement at the beginning of the notebook. You also could perform plane estimation yourself if you are up for a challenge!

# In[72]:


# Graded Function: merge_lane_lines
def merge_lane_lines(
        lines):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    
    ### START CODE HERE ### (≈ 25 lines in total)
    
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    
    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)
    
    # Step 2: Determine lines with slope less than horizontal slope threshold.
    temp_s = []
    temp_in = []
    final_s = []
    final_in = []
    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):
        if len(temp_s) == 0:
            temp_s.append(slope)
            temp_in.append(intercept)
        else:
            for i in range(len(temp_s)):
                if np.absolute(slope-temp_s[i])>=slope_similarity_threshold and np.absolute(intercept-temp_in[i])>=intercept_similarity_threshold:
                    final_s.append(slope)
                    final_in.append(intercept)
                else:
                    temp_s.append(slope)
                    temp_in.append(intercept)
                    
    # Step 4: Merge all lines in clusters using mean averaging
    final_s.append(temp_s[0])
    final_in.append(temp_in[0])
    final_lines = []
    y_max = dataset_handler.image.shape[0]
    y_min = np.min(np.argwhere(road_mask == 1)[:, 0])
    
    for slope, intercept, in zip(final_s, final_in):
        x1 = (y_min - intercept) / slope
        x2 = (y_max - intercept) / slope
        final_lines.append([x1, y_min, x2, y_max])

    return np.array(final_lines)
    # Note: Make sure dimensions of returned lines is (N x 4)
    ### END CODE HERE ###


# In[73]:


merged_lane_lines = merge_lane_lines(lane_lines)

plt.imshow(dataset_handler.vis_lanes(merged_lane_lines))


# ***Expected Output***
# 
# <img src="images/lanes_2.png" style="width:320px;height:240px;"> 

# You now should have one line per lane as an output! The final step is to extrapolate the lanes to start at the beginning of the road, and end at the end of the road, and to determine the lane markings belonging to the current lane. We provide you with functions that perform these tasks in the cell bellow. Run the cell to visualize the final lane boundaries!

# In[74]:


max_y = dataset_handler.image.shape[0]
min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)
plt.imshow(dataset_handler.vis_lanes(final_lanes))


# ***Expected Output***
# 
# <img src="images/lanes_final.png" style="width:320px;height:240px;"> 

# ## 3 - Computing Minimum Distance To Impact Using The Output of 2D Object Detection.
# 
# Your final task for this assessment is to use 2D object detection output to determine the minimum distance to impact with objects in the scene. However, the task is complicated by the fact that the provided 2D detections are from a high recall, low precision 2D object detector. You will first be using the semantic segmentation output to determine which bounding boxes are valid. Then, you will compute the minimum distance to impact using the remaining bounding boxes and the depth image. Let us begin with a visualization of the output detection for our current frame. For visualization, you use the provided dataset handler function ``vis_object_detection`` as follows:

# In[75]:


detections = dataset_handler.object_detection

plt.imshow(dataset_handler.vis_object_detection(detections))


# Detections have the format [category, x_min, y_min, x_max, y_max, score]. The Category is a string signifying the classification of the bounding box such as 'Car', 'Pedestrian' or 'Cyclist'. [x_min,y_min] are the coordinates of the top left corner, and [x_max,y_max] are the coordinates of the bottom right corners of the objects. The score signifies the output of the softmax from the neural network.

# In[76]:


print(detections)


# ### 3.1 - Filtering Out Unreliable Detections:
# The first thing you can notice is that an wrong detection occures on the right side of the image. What is interestingis that this wrong detection has a high output score of 0.76 for being a car. Furthermore, two bounding boxes are assigned to the vehicle to the left of the image, both with a very high score, greater than 0.9. This behaviour is expected from a high precision, low recall object detector. To solve this problem, the output of the semantic segmentation network has to be used to eliminate unreliable detections.
# 
# **Exercise**: Eliminate unreliable detections using the output of semantic segmentation. The three steps are:
# 1. For each detection, compute how many pixels in the bounding box belong to the category predicted by the neural network.
# 2. Devide the computed number of pixels by the area of the bounding box (total number of pixels).
# 3. If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
# 
# Usefull functions: ``np.asfarray()``
# 
# ***Note***: Make sure to handle both the 'Car' and 'Pedestrian' categories in the code.

# In[82]:


# Graded Function: filter_detections_by_segmentation
def filter_detections_by_segmentation(detections, segmentation_output):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    segmentation_output -- tensor of dimension (HxW) containing pixel category labels.
    
    Returns:
    filtered_detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    """
    ### START CODE HERE ### (≈ 20 lines in total)
    
    # Set ratio threshold:
    ratio_threshold = 0.3  # If 1/3 of the total pixels belong to the target category, the detection is correct.
    filtered_detections = []
    for detection in detections:
        
        # Step 1: Compute number of pixels belonging to the category for every detection.
        img = segmentation_output[int(np.asfarray(detection[2])):int(np.asfarray(detection[4])),int(np.asfarray(detection[1])):int(np.asfarray(detection[3]))]
        img = np.array(img)
        pix_c = np.zeros(img.shape) #pixel count
        pix_c[img==10] = 1 #vehicle
        # Step 2: Devide the computed number of pixels by the area of the bounding box (total number of pixels).
        p = sum(sum(pix_c))
        q = img.shape[0] * img.shape[1]
        r = p/q
        # Step 3: If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
        if r>=ratio_threshold:
            filtered_detections.append(detection)
 
    ### END CODE HERE ###
    
    return filtered_detections


# In[83]:


filtered_detections = filter_detections_by_segmentation(detections, segmentation)

plt.imshow(dataset_handler.vis_object_detection(filtered_detections))


# ### 3.2 - Estimating Minimum Distance To Impact:
# 
# The final task for this assessment is to estimate the minimum distance to every bounding box in the input detections. This can be performed by simply taking the minimum distance from the pixels in the bounding box to the camera center.
# 
# **Exercise**: Compute the minimum distance to impact between every object remaining after filtering and the self-driving car. The two steps are:
# 
# 1. Compute the distance to the camera center using the x,y,z arrays from  part I. This can be done according to the equation: $ distance = \sqrt{x^2 + y^2 + z^2}$.
# 2. Find the value of the minimum distance of all pixels inside the bounding box.

# In[86]:


# Graded Function: find_min_distance_to_detection:
def find_min_distance_to_detection(detections, x, y, z):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    z -- tensor of dimensions (H,W) containing the z coordinates of every pixel in the camera coordinate frame.
    Returns:
    min_distances -- tensor of dimension (N, 1) containing distance to impact with every object in the scene.

    """
    ### START CODE HERE ### (≈ 20 lines in total)
    min_distances=[]
    for detection in detections:
        # Step 1: Compute distance of every pixel in the detection bounds
        #for all those pixels in the range[min, max]
        d1 = x[int(np.asfarray(detection[2])):int(np.asfarray(detection[4])),
               int(np.asfarray(detection[1])):int(np.asfarray(detection[3]))]
        d2 = y[int(np.asfarray(detection[2])):int(np.asfarray(detection[4])),
               int(np.asfarray(detection[1])):int(np.asfarray(detection[3]))]
        d3 = z[int(np.asfarray(detection[2])):int(np.asfarray(detection[4])),
               int(np.asfarray(detection[1])):int(np.asfarray(detection[3]))]
        dist = np.sqrt(d1**2 + d2**2 + d3**2)
        # Step 2: Find minimum distance
        min_distances.append(dist.min())
    ### END CODE HERE ###
    return min_distances


# In[87]:


min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

print('Minimum distance to impact is: ' + str(min_distances))


# **Expected Output**
# 
# Minimum distance to impact is: 8.51
# 
# 

# Run the cell bellow to visualize your estimated distance along with the 2D detection output.

# In[88]:


font = {'family': 'serif','color': 'red','weight': 'normal','size': 12}

im_out = dataset_handler.vis_object_detection(filtered_detections)

for detection, min_distance in zip(filtered_detections, min_distances):
    bounding_box = np.asfarray(detection[1:5])
    plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m', fontdict=font)

plt.imshow(im_out)


# ## 4 - Submission:
# 
# Evaluation of all the functions will be based on **three** outputs for frame 1 of the dataset:
# 1. The estimated ground plane from part 1.
# 2. The estimated lanes from part 2.
# 3. The estimated distances from part 3. 
# 
# Please run the cell bellow, then copy its output to the provided output.yaml file for submission on the programming assignment page.   

# In[90]:


dataset_handler = DatasetHandler()
dataset_handler.set_frame(1)
segmentation = dataset_handler.segmentation
detections = dataset_handler.object_detection
z = dataset_handler.depth

# Part 1
k = dataset_handler.k
x, y = xy_from_depth(z, k)
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))
p_final = ransac_plane_fit(xyz_ground)

# Part II
lane_lines = estimate_lane_lines(segmentation)
merged_lane_lines = merge_lane_lines(lane_lines)
max_y = dataset_handler.image.shape[0]
min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)

# Part III
filtered_detections = filter_detections_by_segmentation(detections, segmentation)
min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

# Print Submission Info

final_lane_printed = [list(np.round(lane)) for lane in final_lanes]
print('plane:') 
print(list(np.round(p_final, 2)))
print('\n lanes:')
print(final_lane_printed)
print('\n min_distance')
print(list(np.round(min_distances, 2)))


# ### Visualize your Results:
# 
# Make sure your results visualization is appealing before submitting your results.

# In[91]:


# Original Image
plt.imshow(dataset_handler.image)


# In[92]:


# Part I
dist = np.abs(dist_to_plane(p_final, x, y, z))

ground_mask = np.zeros(dist.shape)

ground_mask[dist < 0.1] = 1
ground_mask[dist > 0.1] = 0

plt.imshow(ground_mask)


# In[93]:


# Part II
plt.imshow(dataset_handler.vis_lanes(final_lanes))


# In[94]:


# Part III
font = {'family': 'serif','color': 'red','weight': 'normal','size': 12}

im_out = dataset_handler.vis_object_detection(filtered_detections)

for detection, min_distance in zip(filtered_detections, min_distances):
    bounding_box = np.asfarray(detection[1:5])
    plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m', fontdict=font)

plt.imshow(im_out)


# <font color='blue'>
# **What you should remember**:
# - The output of semantic segmentation can be used to estimate drivable space. 
# - Classical computer vision can be used to find lane boundaries.
# - The output of semantic segmentation can be used to filter out unreliable output from object detection. 

# Congrats on finishing this assignment! 
