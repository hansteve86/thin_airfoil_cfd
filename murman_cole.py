import numpy as np
from scipy.linalg import solve_banded

from .geometry import CircularAirfoil

class MurmanCole():

    def __init__(self, mesh:CircularAirfoil, M_inf:float = 0.9, V_inf:float = 1.0,
                 rho_inf:float = 1.0, gamma:float = 1.4):
#no mach above 0.7
        self.__mesh = mesh
        X, Y = self.__mesh.meshgrid()
        self.X = np.transpose(X)
        self.Y = np.transpose(Y)

        self.phi = np.ones((self.__mesh.Nx, self.__mesh.Ny))
        self.residual = np.zeros_like(self.phi)
        self.u = np.zeros_like(self.phi)
        self.v = np.zeros_like(self.phi)
        self.M = np.zeros_like(self.phi)
        self.p = np.zeros_like(self.phi)
        self.cp = np.zeros_like(self.phi)

        self.M_inf = M_inf
        self.rho_inf = rho_inf
        self.V_inf = V_inf
        self.gamma = gamma
        self.a_inf = V_inf / M_inf
        self.p_inf = 1.0 / (gamma * M_inf**2)

        self.symmetry_bc = np.zeros_like(self.__mesh.x)
        chord_start, chord_stop = self.__mesh.get_indices_of_airfoil()
        self.symmetry_bc[chord_start:chord_stop] = self.__mesh.dy_dx()
        self.dy_min = self.__mesh.y[1] - self.__mesh.y[0]

    def initial_condition(self, levels=2):
        phi = np.ones((levels, self.__mesh.Nx, self.__mesh.Ny))
        phi[0,:,0] = phi[0,:,1] - self.dy_min * self.V_inf * self.symmetry_bc
        return phi

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



    def compute_A_and_mu(self):
        A = np.ones((self.__mesh.Nx, self.__mesh.Ny)) - self.M_inf**2

        dx = np.reshape(self.__mesh.x[1:-1] - self.__mesh.x[:-2], (self.__mesh.Nx-2,1))

        A[1:-1,:] = 1.0 - self.M_inf**2 - (self.gamma + 1) * self.M_inf**2 / self.V_inf * (
            (self.phi[1:-1,:] - self.phi[:-2,:]) / dx
        )

        mu = np.zeros_like(A)
        for j in range(self.__mesh.Ny):
           for i in range(self.__mesh.Nx):
               if A[i,j] < 0.0:
                   mu[i,j] = 1.0

        return A, mu

    def solve(self, print_residuals=1, max_iterations=5000, max_residual=1.0e-5):

        ab = np.zeros((3,self.__mesh.Ny), dtype=np.float64)
        R = np.zeros((self.__mesh.Ny,))

        phi = self.initial_condition()

        residual = 1.0
        iterations = 0
        while residual > max_residual and iterations < max_iterations:

            A, mu = self.compute_A_and_mu()

            for i in range(2,self.__mesh.Nx-1):
                for j in range(1,self.__mesh.Ny-1):

                    dxpi = self.__mesh.x[i+1] - self.__mesh.x[i]
                    dxmi = self.__mesh.x[i] - self.__mesh.x[i-1]
                    dxi = 0.5 * (self.__mesh.x[i+1] - self.__mesh.x[i-1])

                    dxpi_m = self.__mesh.x[i] - self.__mesh.x[i-1]
                    dxmi_m = self.__mesh.x[i-1] - self.__mesh.x[i-2]
                    dxi_m = 0.5 * (self.__mesh.x[i] - self.__mesh.x[i-2])

                    dypi = self.__mesh.y[j+1] - self.__mesh.y[j]
                    dymi = self.__mesh.y[j] - self.__mesh.y[j-1]
                    dyi = 0.5 * (self.__mesh.y[j+1] - self.__mesh.y[j-1])

                    self.residual[i,j] = np.abs(
                        (1.0 - mu[i,j]) * A[i,j] * (
                            phi[0,i+1,j] / (dxi * dxpi) -
                            2 * phi[0,i,j] / (dxpi * dxmi) +
                            phi[0,i-1,j] / (dxi * dxmi)
                        ) + mu[i-1,j] * A[i-1,j] * (
                            phi[0,i,j] / (dxi_m * dxpi_m) -
                            2 * phi[0,i-1,j] / (dxpi_m * dxmi_m) +
                            phi[0,i-2,j] / (dxi_m * dxmi_m)
                        ) +
                            phi[0,i,j+1] / (dyi * dypi) -
                            2 * phi[0,i,j] / (dypi * dymi) +
                            phi[0,i,j-1] / (dyi * dymi)
                    )
            residual = np.max(self.residual)

            # First point
            #
            i = 1

            dxpi = self.__mesh.x[i+1] - self.__mesh.x[i]
            dxmi = self.__mesh.x[i] - self.__mesh.x[i-1]
            dxi = 0.5 * (self.__mesh.x[i+1] - self.__mesh.x[i-1])

            for j in range(1,self.__mesh.Ny-1):
                dypj = self.__mesh.y[j+1] - self.__mesh.y[j]
                dymj = self.__mesh.y[j] - self.__mesh.y[j-1]
                dyj = 0.5 * (self.__mesh.y[j+1] - self.__mesh.y[j-1])

                ab[1,j] = -2.0 * A[i,j] / (dxpi * dxmi) - 2.0 / (dypj * dymj)  # diagonal
                ab[2,j-1] = 1.0 / (dyj * dymj)                          # lower
                ab[0,j+1] = 1.0 / (dyj * dypj)                          # upper

                R[j] = -A[i,j] * (phi[0,i+1,j] / (dxpi * dxi) + phi[0,i-1,j] / (dxmi * dxi))

            # Boundary conditions at j=0
            ab[1,0] = 1.0    # diagonal
            ab[0,1] = -1.0   # upper

            # Boundary conditions at top boundary
            ab[1,-1] = 1.0   # diagonal
            ab[2,-2] = 0.0   # lower

            R[-1] = 1.0
            R[0] = -self.dy_min * self.V_inf * self.symmetry_bc[i]

            phi[1,i,:] = solve_banded((1,1), ab, R)
            phi[1,0,:] = phi[1,i,:]

            for i in range(2,self.__mesh.Nx-1):
                dxpi = self.__mesh.x[i+1] - self.__mesh.x[i]
                dxmi = self.__mesh.x[i] - self.__mesh.x[i-1]
                dxi = 0.5 * (self.__mesh.x[i+1] - self.__mesh.x[i-1])

                dxpi_m = self.__mesh.x[i] - self.__mesh.x[i-1]
                dxmi_m = self.__mesh.x[i-1] - self.__mesh.x[i-2]
                dxi_m = 0.5 * (self.__mesh.x[i] - self.__mesh.x[i-2])

                for j in range(1,self.__mesh.Ny-1):
                    dypj = self.__mesh.y[j+1] - self.__mesh.y[j]
                    dymj = self.__mesh.y[j] - self.__mesh.y[j-1]
                    dyj = 0.5 * (self.__mesh.y[j+1] - self.__mesh.y[j-1])

                    ab[1,j] = -2.0 * (1.0 - mu[i,j]) * A[i,j] / (dxpi * dxmi) + mu[i-1,j] * A[i-1,j] / (dxi_m * dxpi_m) - 2.0 / (dypj * dymj)
                    ab[2,j-1] = 1.0 / (dyj * dymj)   # lower
                    ab[0,j+1] = 1.0 / (dyj * dypj)   # upper

                    R[j] = -(1.0 - mu[i,j]) * A[i,j] * (
                        phi[0,i+1,j] / (dxpi * dxi) + phi[0,i-1,j] / (dxmi * dxi)
                    ) - mu[i-1,j] * A[i-1,j] * (
                        -2.0 * phi[0,i-1,j] / (dxpi_m * dxmi_m) + phi[0,i-2,j] / (dxmi_m * dxi_m)
                    )

                # Boundary conditions at j=0
                ab[1,0] = 1.0    # diagonal
                ab[0,1] = -1.0   # upper

                # Boundary conditions at top boundary
                ab[1,-1] = 1.0   # diagonal
                ab[2,-2] = 0.0   # lower

                R[-1] = 1.0
                R[0] = -self.dy_min * self.V_inf * self.symmetry_bc[i]

                phi[1,i,:] = solve_banded((1,1), ab, R)

            phi[1,0,:] = 1.0
            phi[1,-1,:] = 1.0

            phi[0,:,:] = phi[1,:,:]

            iterations = iterations + 1
            if iterations%print_residuals == 0:
                print(iterations, residual)

        self.phi = phi[0,:,:]
        self.get_solution()
