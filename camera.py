import numpy as np


def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    x = rx*np.pi / 180
    y = ry*np.pi / 180
    z = rz*np.pi / 180
    c = [np.cos(degree) for degree in [x,y,z]]
    s = [np.sin(degree) for degree in [x,y,z]]
    
    matrix = np.zeros((3,3))
    
    X = np.array([[1,   0,  0],
                  [0,c[0],-s[0]],
                  [0,s[0],c[0]]])
    Y = np.array([ [c[1],   0,  -s[1]],
                   [0,1,0],
                   [s[1],0,c[1]]
                 ])
    Z = np.array([
        [c[2],   -s[2],  0],
        [s[2], c[2],0],
        [0, 0, 1]
    ])
    matrix = Z@Y@X
    return matrix

class Camera:
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def project(self,pts3):
        P = np.linalg.inv(self.R)@(pts3 - self.t)
        pts2 = P[:2,:]* self.f/P[2,:]
        pts2 += self.c
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
        return pts2

    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera 
        to use the provided parameters.
        Parameters 
        ----------
        params : 1D numpy.array of shape (6,) (dtype=float)
            Camera parameters we are optimizing over stored in a vector 
            params[:3] are the rotation angles, params[3:] are the translation
        """
        self.R = makerotation(*params[:3])
        self.t = np.array(params[3:]).reshape((3,1))

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'

    