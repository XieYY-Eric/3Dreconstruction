import scipy.optimize
import numpy as np
from PIL import Image
import visutils
import matplotlib.pyplot as plt

def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing stored in a vector of shape (6,)

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
        
    """
    N = pts3.shape[1]
    cam.update_extrinsics(params)
    projections = cam.project(pts3)
    return ((pts2 - projections)).reshape((2*N,))

def calibratePose(pts3,pts2,cam,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera
        
    params_init : 1D numpy.array (dtype=float)
        Initial estimate of camera extrinsic parameters ()
        params[0:3] are the rotation angles, params[3:6] are the translation

    Returns
    -------
    cam : Camera
        Refined estimate of camera with updated R,t parameters
        
    """
    def measure(params):
        return residuals(pts3,pts2,cam,params)
    optimal_p= scipy.optimize.leastsq(measure,params_init)
    cam.update_extrinsics(optimal_p[0])
    return cam

def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """

    #
    # Your code goes here.  I recommend adding assert statements to check the
    # sizes of the inputs and outputs to make sure they are correct
    
    #write the f function
    T = camR.t - camL.t  #translation vector from left camera to right camera 3x1
    n = pts2L.shape[1]
    pts3 = np.zeros((3,n))
    for i in range(n):
        pL,pR = pts2L[:,i].reshape((2,1)),pts2R[:,i].reshape((2,1))
        qL =  np.vstack([(pL-camL.c)/camL.f ,np.ones((1,1))]) #3x1 
        qR =  np.vstack([(pR-camR.c)/camR.f ,np.ones((1,1))]) #3x1
        
        left = camL.R@qL
        right = -camR.R@qR#-R@(np.hstack([pts2R[:,i]/camR.f,1])).reshape((3,1))
        
        A = np.hstack([left,right])
        
        zL,zR = np.linalg.lstsq(A,T,rcond=None)[0] #Z is the depth
        
        PL = qL* zL
        PR = qR* zR
        P = (camL.R@PL + camL.t + camR.R@PR + camR.t)/2
        
        pts3[:,i] = P[:,0]
        

    return pts3

def fileslist(imprefix,start,nums,extension=".jpg"):
    indices = [str(i) for i in range(start,start+nums)]
    if start < 10:
        indices[:10] = ['0'+i for i in indices[:10] if int(i) < 10]
    filename = [imprefix + indice +extension for indice in indices]
    return filename

def decode(filenames,threshold):
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.  Images are assumed to be named "imageprefixN.png" where
    N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")
 
    Parameters
    ----------
    imprefix : str
       Image name prefix
      
    start : int
       Starting index
       
    threshold : float
       Threshold to determine if a bit is decodeable
       
    Returns
    -------
    code : 2D numpy.array (dtype=float)
        Array the same size as input images with entries in (0..1023)
        
    mask : 2D numpy.array (dtype=logical)
        Array indicating which pixels were correctly decoded based on the threshold
    
    """
    BCDs = []
    Masks = []
    img_size = (np.asarray(Image.open(filenames[0])).shape[0],np.asarray(Image.open(filenames[0])).shape[1])
    mask = np.ones(img_size)
    for i in range(0,len(filenames),2):
        up_right_img = np.asarray(Image.open(filenames[i])).astype(float)/256
        if len(up_right_img.shape) == 3:
            up_right_img = up_right_img.mean(2)
            
        invert_img = np.asarray(Image.open(filenames[i+1])).astype(float)/256
        if len(invert_img.shape) == 3:
            invert_img = invert_img.mean(2)
            
        BCD = 1 *(up_right_img > invert_img)
#         mask = np.abs(up_right_img-invert_img) > threshold
        mask = mask * (1*(abs(up_right_img-invert_img)>threshold))
        BCDs.append(BCD)
        Masks.append(mask)
    # we will assume a 10 bit code
    nbits = 10
    code = np.zeros(BCDs[0].shape)
    Bits = [BCDs[0]]
    for i in range(9):
        Bits.append(np.logical_xor(Bits[i],BCDs[i+1]))

    for n in range(nbits):
        code+=Bits[9-n]*(2**n)
    
        
    return code,mask

def reconstruct(leftBits,rightBits,camL,camR):
    """
    Performing matching and triangulation of points on the surface using structured
    illumination. This function decodes the binary graycode patterns, matches 
    pixels with corresponding codes, and triangulates the result.
    
    The returned arrays include 2D and 3D coordinates of only those pixels which
    were triangulated where pts3[:,i] is the 3D coordinte produced by triangulating
    pts2L[:,i] and pts2R[:,i]

    Parameters
    ----------
    imprefixL, imprefixR : str
        Image prefixes for the coded images from the left and right camera
        
    threshold : float
        Threshold to determine if a bit is decodeable
   
    camL,camR : Camera
        Calibration info for the left and right cameras
        
    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)
        The 2D pixel coordinates of the matched pixels in the left and right
        image stored in arrays of shape 2xN
        
    pts3 : 2D numpy.array (dtype=float)
        Triangulated 3D coordinates stored in an array of shape 3xN
        
    """

    # Decode the H and V coordinates for the two views
    
    # Find the indices of pixels in the left and right code image that 
    # have matching codes. If there are multiple matches, just
    # choose one arbitrarily.
    _,matchL,matchR = np.intersect1d(leftBits, rightBits, return_indices=True)
    
    # Let CL and CR be the flattened arrays of codes for the left and right view
    # Suppose you have computed arrays of indices matchL and matchR so that 
    # CL[matchL[i]] == CR[matchR[i]] for all i.  The code below gives one approach
    # to generating the corresponding pixel coordinates for the matched pixels.
    w = rightBits.shape[1]
    h = rightBits.shape[0]
    
    
    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))
    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)

    # Now triangulate the points
    pts3 = triangulate(pts2L,camL,pts2R,camR)
    
    
    return pts2L,pts2R,pts3

def draw_3D(camL,camR,pts3,pts3True=None):
    # generate coordinates of a line segment running from the center
    # of the camera to 3 units in front of the camera
    lookL = np.hstack((camL.t,camL.t+camL.R @ np.array([[0,0,5]]).T))
    lookR = np.hstack((camR.t,camR.t+camR.R @ np.array([[0,0,5]]).T))

    # visualize the left and right image overlaid
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1,projection='3d')
    ax.plot(pts3[0,:],pts3[1,:],pts3[2,:],'.')
    if pts3True:
        ax.plot(pts3True[0,:],pts3True[1,:],pts3True[2,:],'rx')
    ax.plot(camL.t[0],camL.t[1],camL.t[2],'bo') ##left camera blue
    ax.plot(camR.t[0],camR.t[1],camR.t[2],'ro') ##right camera red
    ax.plot(lookL[0,:],lookL[1,:],lookL[2,:],'b')
    ax.plot(lookR[0,:],lookR[1,:],lookR[2,:],'r')
    visutils.set_axes_equal_3d(ax)
    visutils.label_axes(ax)
    ax.set_xlim3d([-20,20])
    ax.set_ylim3d([-20,30])
    ax.set_zlim3d([-100,100])
    plt.title('scene 3D view')

    ax = fig.add_subplot(2,2,2)
    ax.plot(pts3[0,:],pts3[2,:],'.')
    if pts3True:
        ax.plot(pts3True[0,:],pts3True[2,:],'rx')
    ax.plot(camR.t[0],camR.t[2],'ro')
    ax.plot(camL.t[0],camL.t[2],'bo')
    ax.plot(lookL[0,:],lookL[2,:],'b')
    ax.plot(lookR[0,:],lookR[2,:],'r')
    plt.title('XZ-view')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.axis([-10,30,-50,50])


    ax = fig.add_subplot(2,2,3)
    ax.plot(pts3[1,:],pts3[2,:],'.')
    if pts3True:
        ax.plot(pts3True[1,:],pts3True[2,:],'rx')
    ax.plot(camR.t[1],camR.t[2],'ro')
    ax.plot(camL.t[1],camL.t[2],'bo')
    ax.plot(lookL[1,:],lookL[2,:],'b')
    ax.plot(lookR[1,:],lookR[2,:],'r')
    plt.title('YZ-view')
    plt.grid()
    plt.xlabel('y')
    plt.ylabel('z')
    plt.axis([30,-30,-100,100])

    ax = fig.add_subplot(2,2,4)
    ax.plot(pts3[0,:],pts3[1,:],'.')
    if pts3True:
        ax.plot(pts3True[0,:],pts3True[1,:],'rx')
    ax.plot(camR.t[0],camR.t[1],'ro')
    ax.plot(camL.t[0],camL.t[1],'bo')
    ax.plot(lookL[0,:],lookL[1,:],'b')
    ax.plot(lookR[0,:],lookR[1,:],'r')
    plt.title('XY-view')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-20,20,-30,50])