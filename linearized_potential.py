import numpy as np
from scipy.linalg import solve_banded

from .geometry import CircularAirfoil

class EllipticSolver():

    def __init__(self, mesh:CircularAirfoil, M_inf:float = 0.5, p_inf:float = 1.0,
                 rho_inf:float = 1.0, gamma:float = 1.4):

        self.__mesh = mesh
        X, Y = self.__mesh.meshgrid()
        self.X = np.transpose(X)
        self.Y = np.transpose(Y)

        self.phi = np.zeros((self.__mesh.Nx, self.__mesh.Ny))
        self.residual = np.zeros_like(self.phi)
        self.u = np.zeros_like(self.phi)
        self.v = np.zeros_like(self.phi)
        self.M = np.zeros_like(self.phi)
        self.p = np.zeros_like(self.phi)
        self.cp = np.zeros_like(self.phi)

        self.M_inf = M_inf
        self.p_inf = p_inf
        self.rho_inf = rho_inf
        self.gamma = gamma

        self.a_inf = np.sqrt(gamma * p_inf / rho_inf)
        self.V_inf = self.M_inf * self.a_inf

        self.symmetry_bc = np.zeros_like(self.__mesh.x)
        chord_start, chord_stop = self.__mesh.get_indices_of_airfoil()
        self.symmetry_bc[chord_start:chord_stop] = self.__mesh.dy_dx()
        self.phi_inf = self.V_inf * self.__mesh.x
        self.dy_min = self.__mesh.y[1] - self.__mesh.y[0]

    def initial_condition(self, levels=2):
        phi = np.ones((levels, self.__mesh.Nx, self.__mesh.Ny))
        for j in range(self.__mesh.Ny):
            phi[0,:,j] = self.phi_inf[:]
        phi[0,:,0] = phi[0,:,1] - self.dy_min * self.V_inf * self.symmetry_bc
        return phi

    # def get_solution(self):
    #     u = np.zeros((self.__mesh.Nx, self.__mesh.Ny))
    #     v = np.zeros((self.__mesh.Nx, self.__mesh.Ny))

    #     dx = np.reshape(self.__mesh.x[2:] - self.__mesh.x[:-2], (self.__mesh.Nx-2,1))
    #     u[1:-1,:] = (self.phi[2:,:] - self.phi[:-2,:]) / dx
    #     u[0,:] = self.V_inf
    #     u[-1,:] = u[-2,:]

    #     v[:,1:-1] = (self.phi[:,2:] - self.phi[:,:-2]) / (self.__mesh.y[2:] - self.__mesh.y[:-2])
    #     v[:,0] = 0.0
    #     v[:,-1] = v[:,-2]

    #     p = self.p_inf * (1.0 - 0.5 * (self.gamma - 1) * self.M_inf**2 * ((u**2 + v**2) / self.V_inf**2 - 1) )**(self.gamma / (self.gamma - 1))
    #     cp = (p - self.p_inf) / (0.5 * self.rho_inf * self.V_inf**2)

    #     return (u, v, p, cp)

    def get_solution(self):
        dx = np.reshape(self.__mesh.x[1:-1] - self.__mesh.x[:-2], (self.__mesh.Nx-2,1))
        self.u[1:-1,:] = (self.phi[1:-1,:] - self.phi[:-2,:]) / dx + self.V_inf
        self.u[0,:] = self.V_inf
        self.u[-1,:] = self.u[-2,:]

        self.v[:,1:-1] = (self.phi[:,2:] - self.phi[:,:-2]) / (self.__mesh.y[2:] - self.__mesh.y[:-2])
        self.v[:,0] = 0.0
        self.v[:,-1] = self.v[:,-2]

        self.p = self.p_inf * (1.0 - 0.5 * (self.gamma - 1) * self.M_inf**2 * ((self.u**2 + self.v**2) / self.V_inf**2 - 1) )**(self.gamma / (self.gamma - 1))
        self.cp = (self.p - self.p_inf) / (0.5 * self.rho_inf * self.V_inf**2)

        self.M = np.sqrt((self.u**2 + self.v**2)) / self.a_inf


    def solve(self, print_residuals=1, max_iterations=5000, max_residual=1.0e-5):

        A = 1.0 - self.M_inf**2
        phi = self.initial_condition()

        ab = np.zeros((3,self.__mesh.Ny), dtype=np.float64)
        R = np.zeros((self.__mesh.Ny,))

        # Need to re-shape so that we can use numpy broadcasting
        #
        dxp = np.reshape(self.__mesh.x[2:] - self.__mesh.x[1:-1], (self.__mesh.Nx-2,1))
        dxm = np.reshape(self.__mesh.x[1:-1] - self.__mesh.x[:-2], (self.__mesh.Nx-2,1))
        dx = np.reshape(0.5 * (self.__mesh.x[2:] - self.__mesh.x[:-2]), (self.__mesh.Nx-2,1))

        dyp = self.__mesh.y[2:] - self.__mesh.y[1:-1]
        dym = self.__mesh.y[1:-1] - self.__mesh.y[:-2]
        dy = 0.5 * (self.__mesh.y[2:] - self.__mesh.y[:-2])

        residual = 1.0
        iterations = 0
        while residual > max_residual and iterations < max_iterations:

            residual = np.max(np.abs(
                A * (phi[0,2:,1:-1] / (dx * dxp) -
                     2 * phi[0,1:-1,1:-1] / (dxp * dxm) +
                     phi[0,:-2,1:-1] / (dx * dxm)
                ) +
                    phi[0,1:-1,2:] / (dy * dyp) -
                    2 * phi[0,1:-1,1:-1] / (dyp * dym) +
                    phi[0,1:-1,:-2] / (dy * dym)
            ))

            for i in range(1,self.__mesh.Nx-1):
                # dxp, dxm, and dx variables already have the boundary points removed, therefore
                # we must use the 'i-1' indexing to correctly align.
                #
                ab[0,2:]  = -1.0 / (dy * dyp)                                     # upper
                ab[1,1:-1] = 2.0 * A / (dxp[i-1] * dxm[i-1]) + 2.0 / (dyp * dym)  # diagonal
                ab[2,:-2] = -1.0 / (dy * dym)                                     # lower

                R[1:-1] = A * (phi[0,i+1,1:-1] / (dxp[i-1] * dx[i-1]) + phi[0,i-1,1:-1] / (dxm[i-1] * dx[i-1]))

                # Boundary conditions at j=0
                ab[1,0] = 1.0    # diagonal
                ab[0,1] = -1.0   # upper

                # Boundary conditions at top boundary
                ab[1,-1] = 1.0   # diagonal
                ab[2,-2] = 0.0   # lower

                R[-1] = self.phi_inf[i]
                R[0] = -self.dy_min * self.V_inf * self.symmetry_bc[i]

                phi[1,i,:] = solve_banded((1,1), ab, R)
            phi[1,0,:] = self.phi_inf[0]
            phi[1,-1,:] = self.phi_inf[-1]

            phi[0,:,:] = phi[1,:,:]

            iterations = iterations + 1
            if iterations%print_residuals == 0:
                print(iterations, residual)

        self.phi = phi[0,:,:]
        self.get_solution()
