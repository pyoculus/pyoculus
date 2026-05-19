from .cylindrical_bfield import CylindricalBfield
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from scipy.integrate import quad


class AxisymmetricCylindricalGridField(CylindricalBfield):
    """ 
    Axisymmetric magnetic field provided by interpolating a grid of points given in the R-Z plane. 

    Tokamak equilibrium solvers often provide the relevant data in a grid of points in the R-Z plane. 
    """

    def __init__(self, R, Z, B_R, B_Z, B_phi, F_psi, pertfield: CylindricalBfield = None):
        """
        R: numpy array specifying the R coordinates of the grid points
        Z: numpy array specifying the Z coordinates of the grid points
        B_R: numpy array specifying the R component of the magnetic field at each grid point
        B_Z: numpy array specifying the Z component of the magnetic field at each grid point
        B_phi: numpy array specifying the phi component of the magnetic field at each grid point
        """
        super().__init__(1)
        self.R = R
        self.Z = Z
        self.B_phi = B_phi
        self.F_psi = F_psi

        self.F_psi_interpolator = RegularGridInterpolator((R, Z), F_psi, method='quintic')
        self.B_R_derived = lambda xx: -1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[0,1])[0]
        self.B_Z_derived = lambda xx: 1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[1,0])[0]
        self.B_phi_interpolator = RegularGridInterpolator((R, Z), B_phi, method='quintic')
        self.B_R_interpolator = RegularGridInterpolator((R, Z), B_R, method='quintic')
        self.B_Z_interpolator = RegularGridInterpolator((R, Z), B_Z, method='quintic')
        self.pertfield=pertfield
        self.pertamp = 1
        if pertfield is None:
            self.pertfun = lambda xx: np.zeros(3)
        else:
            self.pertfun = pertfield.B


    @classmethod
    def from_matlab_file(cls, filename, with_perturbation=False):
        """
        filename: string specifying the name of the .mat file containing the grid of points. 

        for example, MEQ equilibrium solver calculates the magnetic data in Matlab, the user needs to save the data in a .mat file and provide the filename to this function.
        """
        import scipy.io
        data = scipy.io.loadmat(filename)
        R = data['rr'][0,:]
        Z = data['zz'][:,0]
        B_R = data['Br'].T
        B_Z = data['Bz'].T
        B_phi = data['Bphi'].T
        F_psi = data['Fx'].T
        if with_perturbation:
            F_psi_cosphi = data['Fx_cosphi'].T
            F_psi_sinphi = data['Fx_sinphi'].T
            pertfield = AxisymmetricGridPerturbation(R, Z, F_psi_cosphi, F_psi_sinphi)
            return cls(R, Z, B_R, B_Z, B_phi, F_psi, pertfield=pertfield)
        else:
            return cls(R, Z, B_R, B_Z, B_phi, F_psi)

##### With separatrix current

    @classmethod
    def from_matlab_file_with_added_B(cls, filename, B_R_s, B_Z_s, F_psi_s ,with_perturbation=False):
        """
        filename: string specifying the name of the .mat file containing the grid of points. 

        for example, MEQ equilibrium solver calculates the magnetic data in Matlab, the user needs to save the data in a .mat file and provide the filename to this function.
        """
        import scipy.io 
        data = scipy.io.loadmat(filename)
        R = data['rr'][0,:]
        Z = data['zz'][:,0]
        B_R = data['Br'].T+B_R_s.T
        B_Z = data['Bz'].T+B_Z_s.T
        B_phi = data['Bphi'].T
        F_psi = data['Fx'].T+F_psi_s.T
        if with_perturbation:
            F_psi_cosphi = data['Fx_cosphi'].T
            F_psi_sinphi = data['Fx_sinphi'].T
            pertfield = AxisymmetricGridPerturbation(R, Z, F_psi_cosphi, F_psi_sinphi)
            return cls(R, Z, B_R, B_Z, B_phi, F_psi, pertfield=pertfield)
        else:
            return cls(R, Z, B_R, B_Z, B_phi, F_psi)    
    
    def set_perturbation_amplitude(self, amplitude):
        self.pertamp = amplitude
        self.pertfun = lambda xx: amplitude * self.pertfield.B(xx)

    def B_axi(self, xx):
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        xx2d = xx[::2] # xx is a rphiz vector, only pick R and Z.
        return np.hstack([self.B_R_derived(xx2d), self.B_phi_interpolator(xx2d)[0], self.B_Z_derived(xx2d)])
    
    def B(self, xx):
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        return self.B_axi(xx) + self.pertfun(xx) 

    def B_interpolated(self, xx):
        """
        Evaluate magnetic field directly, for comparison with the derivation of the flux funcitons. 
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        xx2d = xx[::2]
        return np.array([self.B_R_interpolator(xx2d)[0], self.B_phi_interpolator(xx2d)[0], self.B_Z_interpolator(xx2d)[0]])
    

    def A_R(self, xx):
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the vector potential is to be evaluated
        return value of A_R for this specific point
        """
        xx2d = xx[::2]

        def integrand(Zp):
            return self.B_phi_interpolator([xx2d[0],Zp])[0] 
        A_r, err = quad(integrand, self.Z[0], xx2d[1])  

        return A_r



    def A_unperturbed(self, xx): 
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the vector potential is to be evaluated
        """
        xx2d = xx[::2]
        
        return np.array([self.A_R(xx), self.F_psi_interpolator(xx2d)[0]/(2*np.pi*xx2d[0]), 0]) ###AR or AZ=0 gauge ????
    


    def A(self, xx): 
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the vector potential is to be evaluated
        """
        if self.pertfield is None:
            return self.A_unperturbed(xx)
        else:
            return self.A_unperturbed(xx) + self.pertfield.A(xx) * self.pertamp

    def dBdX(self, xx):
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field gradient is to be evaluated
        return Bfield, jacobian matrix for a specific point
        """
        xx2d = xx[::2]
      
        dR=np.array([-1/xx[0]*self.B_R_derived(xx2d)-1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx2d, nu=[1,1])[0],      0    ,  -1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx2d, nu=[0,2])[0]])

        dPhi=np.array([self.B_phi_interpolator(xx2d, nu=[1,0])[0],  0 ,  self.B_phi_interpolator(xx2d, nu=[0,1])[0]])

        dZ=np.array([-1/xx[0]*self.B_Z_derived(xx2d)+1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx2d, nu=[2,0])[0]   ,      0 ,  1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx2d, nu=[1,1])[0]])  

        if self.pertfield is None:
            return self.B(xx), (np.vstack([dR, dPhi, dZ]))
        else:
            return self.B(xx), (np.vstack([dR, dPhi, dZ])+self.pertfield.dBdX(xx)[1]*self.pertamp)




class AxisymmetricGridPerturbation(CylindricalBfield):
    """
    Axisymmetric magnetic field perturbation provided by interpolating a grid of points given in the R-Z plane.
    """
    def __init__(self, R, Z, F_psi_cosphi, F_psi_sinphi):
        """
        Create the perturbation field from the perturbation
        flux function grids.
        """
        super().__init__(1)
        self.R = R
        self.Z = Z
        self.F_psi_cosphi = F_psi_cosphi
        self.F_psi_sinphi = F_psi_sinphi

        self.F_psi_cosphi_interpolator = RegularGridInterpolator((R, Z), F_psi_cosphi, method='quintic')
        self.F_psi_sinphi_interpolator = RegularGridInterpolator((R, Z), F_psi_sinphi, method='quintic')

    def B_R(self, xx):
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        xx2d = xx[::2]
        B_R_cosphi = -1/(2*np.pi*xx[0]) * self.F_psi_cosphi_interpolator(xx2d, nu=[0,1])[0] * np.cos(xx[1])
        B_R_sinphi = -1/(2*np.pi*xx[0]) * self.F_psi_sinphi_interpolator(xx2d, nu=[0,1])[0] * np.sin(xx[1])
        return B_R_cosphi + B_R_sinphi
    
    def B_Z(self, xx):
        xx2d = xx[::2]
        B_Z_cosphi = 1/(2*np.pi*xx[0]) * self.F_psi_cosphi_interpolator(xx2d, nu=[1,0])[0] * np.cos(xx[1])
        B_Z_sinphi = 1/(2*np.pi*xx[0]) * self.F_psi_sinphi_interpolator(xx2d, nu=[1,0])[0] * np.sin(xx[1])
        return B_Z_cosphi + B_Z_sinphi
    
    def B(self, xx): 
        return np.hstack([self.B_R(xx), 0, self.B_Z(xx)])
    
    def A(self, xx):
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic potential is to be evaluated
        """
        xx2d = xx[::2]
        
        return np.array([0 , 1/(2*np.pi*xx[0]) * (self.F_psi_cosphi_interpolator(xx2d)[0] * np.cos(xx[1]) + self.F_psi_sinphi_interpolator(xx2d)[0] * np.sin(xx[1])) ,0])
    
    def dBdX(self, xx, *args):

        """Gradient of the total field function at the point coords. Where (dBdX)^i_j = dB^i/dX^j with i the row index and j the column index of the matrix."""

        xx2d = xx[::2]


        dR = np.array([-1/xx[0]*self.B_R(xx)  -   1/(2*np.pi*xx[0]) * (self.F_psi_cosphi_interpolator(xx2d, nu=[1,1])[0] * np.cos(xx[1]) + self.F_psi_sinphi_interpolator(xx2d, nu=[1,1])[0]*np.sin(xx[1])), 
                            1/(2*np.pi*xx[0]) * (self.F_psi_cosphi_interpolator(xx2d, nu=[0,1])[0] * np.sin(xx[1]) - self.F_psi_sinphi_interpolator(xx2d, nu=[0,1])[0] * np.cos(xx[1])),
                           -1/(2*np.pi*xx[0]) * (self.F_psi_cosphi_interpolator(xx2d, nu=[0,2])[0] * np.cos(xx[1]) + self.F_psi_sinphi_interpolator(xx2d, nu=[0,2])[0] * np.sin(xx[1]))])
                           
        dPhi = np.array([0,0,0])
        
        dZ = np.array([-1/xx[0]*self.B_Z(xx)  +   1/(2*np.pi*xx[0]) * (self.F_psi_sinphi_interpolator(xx2d, nu=[2,0])[0] * np.cos(xx[1])+np.sin(xx[1])*self.F_psi_sinphi_interpolator(xx2d, nu=[2,0])[0]), 
                           -1/(2*np.pi*xx[0]) * (self.F_psi_cosphi_interpolator(xx2d, nu=[1,0])[0] * np.sin(xx[1]) - self.F_psi_sinphi_interpolator(xx2d, nu=[1,0])[0] * np.cos(xx[1])),
                            1/(2*np.pi*xx[0]) * (self.F_psi_cosphi_interpolator(xx2d, nu=[1,1])[0] * np.cos(xx[1]) + self.F_psi_sinphi_interpolator(xx2d, nu=[1,1])[0] * np.sin(xx[1]))])
        
        return self.B(xx), np.vstack([dR,dPhi,dZ]) 

