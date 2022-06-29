import numpy as np

class CircularAirfoil:
    """Define a two-dimensional mesh for a thin, circular airfoil. The airfoil
    is assumed to be thin enough such that the boundary conditions can be applied
    directly on the symmetry line (y = 0) and not on the actual airfoil surface
    itself. Therefore, the mesh is *not* body-fitted.

    Args:
        thickness (float): Thickness of the circular arc airfoil in percent. Defaults
            to 6, i.e, a 6% thick circular air airfoil.
        c (float, optional): Chord length of the airfoil. Defaults to 1.0.
        Nx (int, optional): Total number of mesh points to use in the x-direction.
            Defaults to 51.
        Ny (int, optional): Total number of mesh points to use in the y-direction.
            Defaults to 51.
        freestream_fraction (float, optional): The precent of mesh points to allocate
            for the freestream flow before and after the airfoil. Defaults to 0.3.
        stretch_mesh (bool, optional): Exponentially stretch the mesh in the freestream.
            Defaults to True.
    """
    def __init__(self, thickness:float = 6 , c:float = 1.0, Nx:int = 51, Ny:int = 51,
                 freestream_fraction=0.3, stretch_mesh:bool = True):
        self.Nx = Nx
        self.Ny = Ny
        self.c = c
        self.thickness = thickness / 100.0

        self.freestream_fraction = freestream_fraction
        self.__FREESTREAM_LENGTH = 50 # Multiple of chord length

        # Length of domain in x-direction composed of the chord-length
        # plus freestream lengths in front and behind the airfoil.
        #
        self.Lx = (2.0 * self.__FREESTREAM_LENGTH + 1) * c
        self.Ly = self.__FREESTREAM_LENGTH * c

        self.x = np.linspace(-self.__FREESTREAM_LENGTH, self.__FREESTREAM_LENGTH + 1, self.Nx)
        self.y = np.linspace(0.0, self.Ly, self.Ny)

        # Create stretched mesh
        if stretch_mesh:
            self._create_stretched_mesh()

    def plot_circular_arc(self):
        """Return the coordinates of the circular arc airfoil. Only used
        for the purposes of plotting.

        Returns:
            (np.ndarray, np.ndarray): Return the (x,y) pair of Cartesian coordinates.
        """

        R = 0.25 * self.c * (self.thickness + 1.0 / self.thickness)

        alpha_start = np.arccos(0.5 * self.c / R)
        alpha_stop = np.pi - alpha_start

        theta = np.linspace(alpha_start, alpha_stop, self.Nx)

        x = R * np.cos(theta)
        y = 0.5 * R * np.sin(theta) - 0.5 * R + 0.5 * self.thickness * self.c
        #y1 = 0.52 + 0.482x + -7.11E-03x^2 + 6.43E-05x^3 + -3.39E-07x^4 + 1.09E-09x^5 + -2.23E-12x^6 + 2.87E-15x^7 + -2.28E-18x^8 + 1.01E-21x^9 + -1.93E-25x^10
        x2 = np.linspace(0, 1000, self.Nx)
        y2 = 0.52 + 0.482*x2 - 7.11*0.001*x2**2 + 6.43*0.00001*x2**3 -3.39*0.0000001*x2**4 + 1.09*0.000000001*x2**5 -2.23*0.00000000000001*x2**6 + 2.87*0.000000000000001*x2**7 -2.28*0.000000000000000001*x2**8 +1.01*0.000000000000000000001*x2**9 - 1.93*0.0000000000000000000000001*x2**10
        return x, y

    def meshgrid(self):
        """Return the (X,Y) coordinates of two-dimensional mesh."""
        return np.meshgrid(self.x, self.y)

    def get_indices_of_airfoil(self):
        """Return the indices marking the beginning and end of the airfoil."""
        Nx_flow = int(self.freestream_fraction * self.Nx)
        Nx_chord = self.Nx - 2 * Nx_flow
        return (Nx_flow, Nx_flow + Nx_chord)

    def dy_dx(self):
        """Return the curvature of the airfoil. Used by the boundary condition."""
        chord_start, chord_stop = self.get_indices_of_airfoil()
        R = 0.25 * self.c * (self.thickness + 1.0 / self.thickness)
        alpha = np.arcsin(0.5 * self.c / R)
        return np.tan(np.linspace(alpha, -alpha, chord_stop - chord_start))
        # return np.linspace(self.thickness / self.c,-self.thickness / self.c, chord_stop - chord_start)

    def _create_stretched_mesh(self):
        """Stretch the mesh exponentially in the freestream flow in the x-, and y-directions.
        Keep a uniform mesh spacing along the chord-length of the airfoil in the x-direction.

        """
        # x-coordinates of mesh
        #
        Nx_flow = int(self.freestream_fraction * self.Nx)
        Nx_chord = self.Nx - 2 * Nx_flow

        Lx_flow = self.__FREESTREAM_LENGTH * self.c
        dx = self.c / (Nx_chord - 1)
        dx_kappa = _stretch(dx, Nx_flow, Lx_flow)

        x_flow = np.linspace(0.0, Lx_flow, Nx_flow)
        for i in range(1,Nx_flow):
            x_flow[i] = x_flow[0] + Lx_flow * (np.exp(dx_kappa * i / (Nx_flow -1)) - 1) / (np.exp(dx_kappa) - 1)

        for i in range(Nx_flow):
            self.x[i] = -x_flow[-1-i] - dx

        for i in range(Nx_flow,Nx_flow+Nx_chord):
            self.x[i] = self.x[i-1] + dx

        for i in range(Nx_flow):
            ii = Nx_flow + Nx_chord
            self.x[ii+i] = self.x[ii-1] + dx + x_flow[i]

        # y-coordinates of mesh
        #
        dy = self.thickness / 10
        dy_kappa = _stretch(dy, self.Ny, self.Ly)

        for i in range(1,self.Ny):
            self.y[i] = self.y[0] + self.Ly * (np.exp(dy_kappa * i / (self.Ny -1)) - 1) / (np.exp(dy_kappa) - 1)


def _stretch(dx_min, N, L, max_error=1.0e-6, max_iterations=50000):
    """The exponential stretching parameter must be computed using Newton's method
    given the starting and ending positions, the minimum spacing, and the total
    number of mesh points. The starting position is assumed to be zero.

    Args:
        dx_min (float): Minimum spacing.
        N (int): Number of mesh points to stretch.
        L (float): Distance over which to stretch the mesh points.
        max_error (float, optional): Error tolerance. Defaults to 1.0e-6.
        max_iterations (int, optional): Maximum number of Netwon iterations. Defaults to 50000.

    Returns:
        float: The exponential stretching parameter.
    """
    k = 1.0

    error = 1.0
    iteration = 0
    while error > max_error and iteration < max_iterations:
        error = L * (np.exp(k / (N -1)) - 1) / (np.exp(k) - 1) - dx_min

        dfdk = L / (np.exp(k) - 1) * (np.exp(k / (N - 1)) / (N - 1) -
                                    np.exp(k) * (np.exp(k / (N - 1)) - 1) / (np.exp(k) - 1))
        k -= error / dfdk
        iteration += 1
    return k
