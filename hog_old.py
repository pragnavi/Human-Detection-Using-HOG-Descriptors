# Steps
# 1. convert to grayscale
# 2. compute gradient magnitudes and gradient angle (Prewitts)
# 3. compute HOG features on databse images
# 4. compute HOG features on test image 
# 5. calculate distance between test and database images 
# 6. use 3NN to classify

from PIL import Image
import numpy as np
import math
import sys

'''
Submitted by: 
Pragnavi Ravuluri Sai Durga pr2370
'''
### Convert to grayscale
def convert_to_grayscale(curr_database_image):

    img = Image.open(curr_database_image)

    # Get the dimensions of the image
    width, height = img.size
    #print(width)
    #print(height)
    #print(img.size)

    # Create a new grayscale image
    gray_img = Image.new('L', (width, height))

    # Loop through each pixel in the image and convert to grayscale
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the current pixel
            r, g, b = img.getpixel((x, y))

            # Convert each channel to grayscale using the formula 𝐼 = round(0.299𝑅 + 0.587𝐺 + 0.114𝐵)
            gray_value = round(0.299*r + 0.587*g + 0.114*b)

            # Set the grayscale value of the current pixel in the new grayscale image
            gray_img.putpixel((x, y), gray_value)

    # Save the grayscale image to disk
    #gray_img.save("grayscale_image.png")

    # Alternatively, display the grayscale image on the screen
   

    return gray_img

### Compute Gradients
def compute_gradients(bw_img):

    img = np.array(bw_img)
    img_h, img_w = img.shape

    sobel_operator_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_operator_y= np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    mask_size = sobel_operator_x.shape[0]

    buffer = mask_size // 2

    g_x = np.zeros((img_h,img_w))
    g_y = np.zeros((img_h,img_w))

    gradient_magnitude = np.zeros((img_h,img_w), dtype=np.float32)
    gradient_angle = np.zeros((img_h,img_w), dtype=np.float32)

    for i in range(0+buffer,img_h-buffer,1):
        for j in range(0+buffer,img_w-buffer,1):
            g_x[i][j] = np.sum(np.multiply(sobel_operator_x,img[i-buffer:i+buffer+1,j-buffer:j+buffer+1]))
            g_y[i][j] = np.sum(np.multiply(sobel_operator_y,img[i-buffer:i+buffer+1,j-buffer:j+buffer+1]))
            gradient_magnitude[i][j] = round(math.sqrt(g_x[i][j]**2 + g_y[i][j]**2))

            theta = np.arctan2(g_y[i][j], g_x[i][j])
            theta_deg = np.degrees(theta)
            theta_deg_mod = np.mod(theta_deg, 360)
            gradient_angle[i][j]= round(theta_deg_mod)
        
    # normalize and round off gradient magnitude to intergers in range [0-255]
    nmz = np.max(gradient_magnitude)/255
    gradient_magnitude = gradient_magnitude/nmz

    return gradient_magnitude,gradient_angle

def HOG(gradient_magnitude, gradient_angle):
    
    #Creating a feature vector which will be a 3d array
    feature_vector_3d = np.zeros((20,12,9))
    kk = np.zeros(9)
    
    # the final hog vector will have 19*11(block)*36(feature vector)=7524 values
    final_hog_vector = np.zeros(7524)
    block = np.zeros(36)
    
    # Calculating the feature vector values for (160/8=)20 * (96/8)12 = 240 cells
    for i in range(0,19,1):
        for j in range(0,11,1):
            for ii in range(0+8*i,7+8*i,1):
                for jj in range(0+8*j,7+8*j,1):
                    ga = gradient_angle[ii][jj]
                    gm = gradient_magnitude[ii][jj]
                    
                    # Assigning appropriate magnitude to correct bins
                    if 10<=ga<30:
                        kk[0] += (1-(abs(ga-10)/20))*gm
                        kk[1] += (1-(abs(ga-30)/20))*gm
                    if 30<=ga<50:
                        kk[1] += (1-(abs(ga-30)/20))*gm
                        kk[2] += (1-(abs(ga-50)/20))*gm
                    if 50<=ga<70:
                        kk[2] += (1-(abs(ga-50)/20))*gm
                        kk[3] += (1-(abs(ga-70)/20))*gm
                    if 70<=ga<90:
                        kk[3] += (1-(abs(ga-70)/20))*gm
                        kk[4] += (1-(abs(ga-90)/20))*gm
                    if 90<=ga<110:
                        kk[4] += (1-(abs(ga-90)/20))*gm
                        kk[5] += (1-(abs(ga-110)/20))*gm
                    if 110<=ga<130:
                        kk[5] += (1-(abs(ga-110)/20))*gm
                        kk[6] += (1-(abs(ga-130)/20))*gm
                    if 130<=ga<150:
                        kk[6] += (1-(abs(ga-130)/20))*gm
                        kk[7] += (1-(abs(ga-150)/20))*gm
                    if 150<=ga<170:
                        kk[7] += (1-(abs(ga-150)/20))*gm
                        kk[8] += (1-(abs(ga-170)/20))*gm
                    if 170<=ga<180 or 0<=ga<10:
                        kk[8] += (1-(abs(ga-170)/20))*gm
                        kk[0] += (1-(abs(ga-10)/20))*gm

            for k in range(0,8,1):
                feature_vector_3d[i][j][k] = kk[k]
                kk[k]=0

    count = 0

    # Calculating the final hog feature vector. 
    # There are 19*11 blocks due to one cell overlap
    for i in range(0,18,1):
        for j in range(0,10,1):
            square_sum = 0
            for k in range(0,8,1):
                block[k] = feature_vector_3d[i][j][k]
                square_sum += block[k]*block[k]
            for k in range(9,17,1):
                block[k] = feature_vector_3d[i][j+1][k-9]
                square_sum += block[k]*block[k]
            for k in range(18,26,1):
                block[k] = feature_vector_3d[i+1][j][k-18]
                square_sum += block[k]*block[k]
            for k in range(27,35,1):
                block[k] = feature_vector_3d[i+1][j+1][k-27]
                square_sum += block[k]*block[k]
            
            count += 36
            block_norm = math.sqrt(square_sum)
            
            if block_norm != 0:
                # Performing L2 normalization
                block = block/block_norm  
                
            c = count-36
            for k in range(c,c+35,1):
                final_hog_vector[k] = block[k%36]
         
    return final_hog_vector

### Calculate similarity

def calculate_similarity(database_HOG, test_HOG):
    height, _ = database_HOG.shape
    hs_similarity = np.zeros((1,height))
    hd_similarity = np.zeros((1,height))

    # iterating through each database image index i
    for i in range(height):
        hs_similarity[0,i] = 1-(np.sum(np.minimum(database_HOG[i,:],test_HOG)))
        hd_similarity[0,i] = 1-(np.sum(math.sqrt(np.matmul(database_HOG[i,:],test_HOG))))
    return hs_similarity, hd_similarity

### Determine 3NN
def threeNN(similarity, n_neg_database):
    neg = 0
    pos = 0
    NN_count = 1

    # pick three largest 
    indices = np.argsort(similarity)[0][-3:]

    for i in indices:
        if i < n_neg_database:
            neg += 1
            print("\nNN #%d: %s, %f, Not-human" % (NN_count, neg_database_files[i], similarity[0][i]))
        else:
            pos +=1
            print("\nNN #%d: %s, %f, Human" % (NN_count, pos_database_files[i%n_neg_database], similarity[0][i]))
        NN_count += 1

    if neg > pos:
        classification = 'Not human'
    else:
        classification = 'Human'

    return classification

# main 
# ensure proper arguments given i.e. 
# python3 hog.py "./Test images (Pos)/T1.bmp"
# python3 hog.py "./Test images (Pos)/T2.bmp"
# python3 hog.py "./Test images (Pos)/T3.bmp"
# python3 hog.py "./Test images (Pos)/T4.bmp"
# python3 hog.py "./Test images (Pos)/T5.bmp"
# python3 hog.py "./Test images (Neg)/T6.bmp"
# python3 hog.py "./Test images (Neg)/T7.bmp"
# python3 hog.py "./Test images (Neg)/T8.bmp"
# python3 hog.py "./Test images (Neg)/T9.bmp"
# python3 hog.py "./Test images (Neg)/T10.bmp"

if (len(sys.argv)) < 2:
    print("Command failure. Please pass image path+name as parameter in single quotesand try again.\
        \nExample: $ python3 hog.py './Test images (Neg)/T10.bmp'")
    exit()

# compute HOG on all database images first

# Collect database files
pos_database_files = ['DB1.bmp', 'DB2.bmp', 'DB3.bmp', 'DB4.bmp', 'DB5.bmp', 'DB6.bmp',\
    'DB7.bmp', 'DB8.bmp', 'DB9.bmp', 'DB10.bmp']
neg_database_files = ['DB11.bmp', 'DB12.bmp', 'DB13.bmp', 'DB14.bmp', 'DB15.bmp', 'DB16.bmp',\
    'DB17.bmp', 'DB18.bmp', 'DB19.bmp', 'DB20.bmp']

pos_database_loc = './Database images (Pos)/'
neg_database_loc = './Database images (Neg)/'

n_pos_database = len(pos_database_files)
n_neg_database = len(neg_database_files)
n_database = n_pos_database + n_neg_database


database_HOG = np.zeros((n_database,7524))

# For all positive database images, 
for i, _ in enumerate(pos_database_files):
    # Parse file location for this image
    curr_database_image = pos_database_loc + pos_database_files[i]

    # Show this image
    # im_show = Image.open(curr_database_image)
    # im_show.show()

    # Convert this image to grayscale
    bw_img = convert_to_grayscale(curr_database_image)

    # Compute gradients for this image
    gradient_magnitude, gradient_angle = compute_gradients(bw_img)

    # Compute HOG for this image
    database_HOG[i] = HOG(gradient_magnitude, gradient_angle)

print ("Grayscale, Gradients and HOG for Pos database images complete")

# Now we repeat for all negative database images, 
for i, _ in enumerate(neg_database_files):
    # Parse file location for this image
    curr_database_image = neg_database_loc + neg_database_files[i]

    # Show this image
    # im_show = Image.open(curr_database_image)
    # im_show.show()

    # Convert this image to grayscale
    bw_img = convert_to_grayscale(curr_database_image)

    # Compute gradients for this image
    gradient_magnitude, gradient_angle = compute_gradients(bw_img)

    # Compute HOG for this image
    database_HOG[i+n_pos_database] = HOG(gradient_magnitude, gradient_angle)

print ("Grayscale, Gradients and HOG for Neg database images complete")



# compute HOG on given test image

# show input image from parameter passed
test_ip_image = sys.argv[1]
im_show = Image.open(test_ip_image)
im_show.show()

# Convert this image to grayscale
bw_img = convert_to_grayscale(test_ip_image)

# Compute gradients for this image
gradient_magnitude, gradient_angle = compute_gradients(bw_img)

# Compute HOG for this image
test_HOG = HOG(gradient_magnitude, gradient_angle)

print ("Grayscale, Gradients and HOG for test image complete")

# Calculate distance
hs_similarity, hd_similarity = calculate_similarity(database_HOG, test_HOG)

print ("Distance calculation complete")

# Classify with 3NN (histogram intersection similarity)
similarity = hs_similarity
classification = threeNN(similarity, n_neg_database)
print("\nHistogram Intersection:")
print("\nThe test image is classified as: ", classification)

# Classify with 3NN (hellinger similarity)
similarity = hd_similarity
classification = threeNN(similarity, n_neg_database)

print("\nHellinger Distance:")
print("\nThe test image is classified as: ", classification)