import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BarycentricInterpolator as scibary


class Quadrature:
    """
    This class defines a quadrature method, which is a way to
    approximate the area under a curve of a function of 1 variable.
    It uses the general formula integral(f(x)) = sum(c_i*f(x_i)).
    The function, f(x), should be continuous on the interval of the
    limits of integrations.

    It is meant only as a superclass of a Quadrature rule, such as
    -LeftPointRiemman
    -RightPointRiemman
    -Trapezoidal
    -MidPointRiemman
    -Simpsons
    -Gaussian
    So there is no reason to ever make a direct instance of the
    quadrature class. Thus, it is an abstract class.
    """

    def __init__(self, a, b, n):
        """
        Initialize the left-endpoint: a, the right endpoint: b,
        and the number of intervals: n. Also creates the length of the
        interval: h.

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        self.a = a
        self.b = b
        self.n = n
        self.h = float((b - a)) / n

    def integrate(self, f):
        """
        Integrates the function, f, by means of a quadrature method.
        That is, it is computed by sum(c*f(x)).

        Parameters
        ----------
        f : a function of one variable

        Returns
        -------
        approx : The approximate integral.

        Notes
        -----
        If all the weights are one, it is best to instead set them
        equal to None and this method will skip the wasted
        multiplication.
        """
        # Start timing the approximation.
        start = time.time()

        # Evaluate the quadrature method.
        if (self.weights is None):
            self.approx = self.outer * np.sum(f(self.evaluation_pts))
        else:
            self.approx = self.outer * np.sum(self.weights *
                                              f(self.evaluation_pts))

        # Finish timing the approximation.
        end = time.time()

        # Store and return the approximated integral and the time it
        #   took to run the quadrature method.
        self.time_taken = end - start
        return self.approx

    def error(self, F):
        """
        Figures out the absolute error of the approximated integral.

        Parameters
        ----------
        F : a function of one variable that is the antiderivative of
            the function that is being approximated.

        Returns
        -------
        The absolute error.

        Notes
        -----
        This method must be called AFTER integrate(f).
        """
        return abs((F(self.b) - F(self.a)) - self.approx)


class LeftPointRiemman(Quadrature):
    """
    This class defines the quadrature method Left End-Point Riemman
    summation.

    It works by creating n rectangles of height f(x_i), where
    x_i = a + i*h for i = range(n) and h = (b-a)/n and a,b are the
    limits of integration, to approximate the area under the curve.
    It approximates any polynomial of degree 0 perfectly.

    LeftPointRiemman creates arrays of the evaluation points and
    coefficients needed to perform the approximation. The integrate
    and error methods are performed in its superclass, Quadrature.

    >>> leftLimit = 2
    >>> rightLimit = 4
    >>> numRects = 4
    >>> Q = LeftPointRiemman(leftLimit, rightLimit, numRects)
    >>> f = lambda x: 3*x
    >>> print(Q.integrate(f))
    16.5
    >>> F = lambda x: 3*x**2/2.0
    >>> print(Q.error(F))
    1.5
    """
    def __init__(self, a, b, n):
        """
        Initialize the left-endpoint: a, the right-endpoint: b, and the
        number of intervals: n. Left-Point Riemman summations follow
        the quadrature formula h*sum(f(x_i)) where x_i = a + i*h from
        i = [0..n-1].

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self, a, b, n)
        self.evaluation_pts = np.linspace(a, b - self.h, n)
        self.weights = None
        self.outer = self.h

    def shade_under_curve(self, fig, f):
        """
        shade_under_curve adds shaded in rectangles evaluationed at the
        left end-point to a figure and draws a plot of the function: f.

        Parameters
        ----------
        fig : The figure in which to add the rectangles.
        f : The function to be plotted.
        """
        # Add a subplot to the supplied figure.
        ax = fig.add_subplot(111)

        # Get the points to make the rectangles.
        x_points = np.linspace(self.a, self.b, self.n + 1)
        x_points1 = x_points[:-1]
        x_points2 = x_points[1:]
        y_points = f(self.evaluation_pts)

        # Plot the rectangles and shade them in.
        for x1, x2, y in zip(x_points1, x_points2, y_points):
            ax.plot([x1, x1], [0, y], 'y-')
            ax.plot([x2, x2], [0, y], 'y-')
            ax.plot([x1, x2], [y, y], 'y-')
            ax.fill_between([x1, x2], [y, y], color='y', alpha=0.25)

        # Plot the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])


class RightPointRiemman(Quadrature):
    """
    This class defines the quadrature method Right End-Point Riemman
    summation.

    It works by creating n rectangles of height f(x_i), where
    x_i = a + i*h for i = range(1, n+1), h = (b-a)/n, and a,b are the
    limits of integration, to approximate the area under the curve. It
    approximates any polynomial of degree 0 perfectly.

    RightPointRiemman creates arrays of the evaluation points and
    coefficients needed to perform the approximation. The integrate and
    error methods are performed in its superclass, Quadrature.

    >>> leftLimit = -2
    >>> rightLimit = 5
    >>> numRects = 7
    >>> Q = RightPointRiemman(leftLimit, rightLimit, numRects)
    >>> f = lambda x: 2*x + 1
    >>> print(Q.integrate(f))
    35.0
    >>> F = lambda x: x**2 + x
    >>> print(Q.error(F))
    7.0
    """
    def __init__(self, a, b, n):
        """
        Initialize the left-endpoint: a, the right-endpoint: b, and the
        number of intervals: n. Right-Point Riemman summations follow
        the quadrature formula h*sum(f(x_i)) where x_i = a + i*h from
        i = [1..n].

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self, a, b, n)
        self.evaluation_pts = np.array(np.linspace(a + self.h, b, n))
        self.weights = None
        self.outer = self.h

    def shade_under_curve(self, fig, f):
        """
        shade_under_curve adds shaded in rectangles evaluationed at the
        right end-point to a figure and draws a plot of the
        function: f.

        Parameters
        ----------
        fig : The figure in which to add the rectangles.
        f : The function to be plotted.
        """
        # Add a subplot to the supplied figure.
        ax = fig.add_subplot(111)

        # Get the points to make the rectangles.
        x_points = np.linspace(self.a, self.b, self.n + 1)
        x_points1 = x_points[:-1]
        x_points2 = x_points[1:]
        y_points = f(self.evaluation_pts)

        # Plot the rectangles and shade them in.
        for x1, x2, y in zip(x_points1, x_points2, y_points):
            ax.plot([x1, x1], [0, y], 'y-')
            ax.plot([x2, x2], [0, y], 'y-')
            ax.plot([x1, x2], [y, y], 'y-')
            ax.fill_between([x1, x2], [y, y], color='y', alpha=0.25)

        # Plot the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])


class MidPointRiemman(Quadrature):
    """
    This class defines the quadrature method Midpoint Riemman
    summation.

    It creates n rectangles of height f(x_i), where x_i = a + h/2 + i*h
    for i = range(n) and h = (b-a)/n and a,b are the limits of
    integration, to approximate the area under the curve. It
    approximates any polynomial of degree 1 perfectly.

    MidPointRiemman creates arrays of the evaluation points and
    coefficients needed to perform the approximation. The integrate and
    error methods are performed in its superclass, Quadrature.

    >>> leftLimit = -2
    >>> rightLimit = 3
    >>> numRects = 10
    >>> Q = MidPointRiemman(leftLimit, rightLimit, numRects)
    >>> f = lambda x: 4*x + 6
    >>> print(Q.integrate(f))
    40.0
    >>> F = lambda x: 2*x**2 + 6*x
    >>> print(Q.error(F))
    0.0
    """

    def __init__(self, a, b, n):
        """
        Initialize the left-endpoint: a, the right endpoint: b, and the
        number of intervals: n. Mid-Point Riemman Summation follow the
        quadrature formula h*sum(f(x_i)) where x_i = a + h/2 + i*h from
        i = [0..n-1].

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self, a, b, n)
        self.evaluation_pts = np.array(np.linspace(a + self.h / 2.0,
                                       b - self.h / 2.0, n))
        self.weights = None
        self.outer = self.h

    def shade_under_curve(self, fig, f):
        """
        shade_under_curve adds shaded in rectangles evaluationed at the
        mid point to a figure and draws a plot of the function: f.

        Parameters
        ----------
        fig : The figure in which to add the trapezoids or rectangles.
        f : The function to be plotted.
        """
        # Add a subplot to the supplied figure.
        ax = fig.add_subplot(111)

        # Get the points to make the rectangles.
        x_points = np.linspace(self.a, self.b, self.n + 1)
        x_points1 = x_points[:-1]
        x_points2 = x_points[1:]
        y_points = f(self.evaluation_pts)

        # Plot the rectangles and shade them in.
        for x1, x2, y in zip(x_points1, x_points2, y_points):
            ax.plot([x1, x1], [0, y], 'y-')
            ax.plot([x2, x2], [0, y], 'y-')
            ax.plot([x1, x2], [y, y], 'y-')
            ax.fill_between([x1, x2], [y, y], color='y', alpha=0.25)

        # Plot the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])


class Trapezoidal(Quadrature):
    """
    This class defines the quadrature method Composite Trapezoidal
    Rule.

    It works by creating n trapezoids of height f(x_i), where
    x_i = a + i*h for i = range(n+1) and h = (b-a)/n and a,b are the
    limits of integration, to approximate the area under the curve. It
    approximates any polynomial of degree 1 perfectly.

    Trapezoidal creates arrays of the evaluation points and
    coefficients needed to perform the approximation. The integrate
    and error methods are performed in its superclass, Quadrature.
    >>> leftLimit = -1
    >>> rightLimit = 8
    >>> numTraps = 9
    >>> Q = Trapezoidal(leftLimit, rightLimit, numTraps)
    >>> f = lambda x: 3*x**2 + 1
    >>> print(Q.integrate(f))
    526.5
    >>> F = lambda x: x**3 + x
    >>> print(Q.error(F))
    4.5
    """
    def __init__(self, a, b, n):
        """
        Initialize the left-endpoint: a, the right endpoint: b, and the
        number of intervals: n. The Composite Trapezoidal Rule follows
        the quadrature formula h/2*(f(a) + sum(2*f(x_i)) + f(b)) where
        x_i = a + i*h from i = [1..n-1].

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self, a, b, n)
        self.evaluation_pts = np.linspace(a, b, n + 1)
        self.weights = np.concatenate([[1], np.zeros(n - 1) + 2, [1]])
        self.outer = self.h / 2.0

    def shade_under_curve(self, fig, f):
        """
        shade_under_curve adds Shaded in Trapezoids to a figure from
        adjacent evaluation points and draws a plot of the function: f.

        Parameters
        ----------
        fig : The figure in which to add the trapezoids or rectangles.
        f : The function to be plotted.
        """
        # Add a subplot to the supplied figure.
        ax = fig.add_subplot(111)

        # Get the points to make the trapezoids.
        x_points = self.evaluation_pts
        y_points = f(x_points)
        plotPts = zip(x_points, y_points)
        pts1 = plotPts[:-1]
        pts2 = plotPts[1:]

        # Plot the trapezoids and shade them in.
        for pt1, pt2 in zip(pts1, pts2):
            x1, y1 = pt1[0], pt1[1]
            x2, y2 = pt2[0], pt2[1]
            ax.plot([x1, x1], [0, y1], 'y-')
            ax.plot([x2, x2], [0, y2], 'y-')
            ax.plot([x1, x2], [y1, y2], 'y-')
            ax.fill_between([x1, x2], [y1, y2], color='y', alpha=0.25)

        # Plot the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])


class Simpsons(Quadrature):
    """
    This class defines the quadrature method Composite Simpsons Rule.

    It creates n/2 Quadratic fits at 3 adjacent values of
    (x_i, f(x_i)), where x_i = a + i*h for i = range(n+1) and
    h = (b-a)/n and a,b are the limits of integration, and integrating
    them to approximate the area under the curve. It approximates any
    polynomial of degree 3 perfectly.

    Simpsons creates arrays of the evaluation points and coefficients
    needed to perform the approximation. The integrate and error
    methods are performed in its superclass, Quadrature.
    >>> leftLimit = 2
    >>> rightLimit = 11
    >>> numQuadFits = 2
    >>> Q = Simpsons(leftLimit, rightLimit, 2*numQuadFits)
    >>> f = lambda x: 8*x**3 + 6*x**2 + 4*x + 5
    >>> print(Q.integrate(f))
    32175.0
    >>> F = lambda x: 2*x**4 + 2*x**3 + 2*x**2 + 5*x
    >>> print(Q.error(F))
    0.0
    """
    def __init__(self, a, b, n):
        """
        Initialize the left-endpoint: a, the right endpoint: b, and the
        number of intervals: n. The Composite Simpsons Rule follows the
        quadrature formula h/3*(f(a) + sum(4*f(x_i)) + sum(2*f(x_j) +
        f(b)) where x_i = a + (2*i-1)*h from i = [1..floor(n/2)] and
        x_j = a + 2*j*h from j = [1..floor(n/2)-1].

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self, a, b, n)
        self.evaluation_pts = np.linspace(a, b, n + 1)
        self.weights = np.concatenate((np.array([1]),
                                      np.arange(2, 2 * n, 2) % 4 + 2,
                                      np.array([1])))
        self.outer = self.h / 3.0

    def shade_under_curve(self, fig, f):
        """
        Adds floor(n/2) quadratic lagrange interpolants using 3
        adjacent points to fig and shades them underneath. Also draws
        a plot of f.

        Parameters
        ----------
        fig : The figure in which to add the trapezoids or rectangles.
        f : The function to be plotted.
        """
        # Add a sub plot to the supplied figure.
        ax = fig.add_subplot(111)

        # Create the points to make the Quadratic fits.
        x_points = self.evaluation_pts
        y_points = f(x_points)
        points = zip(x_points, y_points)
        pts1 = points[0:-1:2]
        pts2 = points[1::2]
        pts3 = points[2::2]
        plotPts = zip(pts1, pts2, pts3)

        # Plot the Quadratic fits and shade them in.
        for pts in plotPts:
            x_pts = [p[0] for p in pts]
            y_pts = [p[1] for p in pts]
            poly_fit = scibary(x_pts, y_pts)
            x1 = x_pts[0]
            x2 = x_pts[-1]
            if self.n > 150:
                xL = np.linspace(x1, x2, 1)
            else:
                xL = np.linspace(x1, x2, 150 / self.n)
            yL = poly_fit(xL)
            for x, y in pts:
                ax.plot([x, x], [0, y], 'y-')
            ax.plot(xL, yL, 'y-')
            ax.fill_between(xL, yL, color='y', alpha='0.25')

        # Plot the actual function.
        xL2 = np.linspace(self.a, self.b, 150)
        ax.plot(xL2, f(xL2))
        ax.set_xlim([self.a, self.b])


class Gaussian(Quadrature):
    """
    This class defines the quadrature method Gaussian-Legendre
    Quadrature.

    It works by optimally picking the nodes x_i as the roots of the nth
    Legendre Polynomials.

    Guassian creates arrays of the evaluation points and coefficients
    needed to perform the approximation. The integrate and error
    methods are performed in its superclass, Quadrature.
    >>> leftLimit = 2
    >>> rightLimit = 4
    >>> numNodes = 4
    >>> Q = Gaussian(leftLimit, rightLimit, numNodes)
    >>> f = lambda x: 8*x**7 - 6*x**5 + 5
    >>> print(Q.integrate(f))
    61258.0
    >>> F = lambda x: x**8 - x**6 + 5*x
    >>> E = Q.error(F)
    >>> abs(E) <= 1e-9
    True
    """
    def __init__(self, a, b, n):
        """
        Initialize the left-endpoint: a, the right endpoint: b, and the
        number of intervals: n. The Gaussian-Legendre method follows
        the quadrature formula (b - a)/2*sum(c_i*f((b - a)/2*x_i +
        (b + a)/2)) where x_i are the roots of the nth legendre
        polynomials and c_i is the associated lagrange interpolant
        weight.

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self, a, b, n)

        # Create the terms for transforming the interval from
        # [-1,1] -> [a,b]
        term1 = (b - a) / 2.0
        term2 = (b + a) / 2.0
        self.outer = term1

        # Big list of the roots of the legendre polynomials up to 16.
        big_roots = [[0.0],
                    [0.577350269189626],
                    [0.774596669241483, 0.000000000000000],
                    [0.86113631159405, 0.339981043584856],
                    [0.90617984593866,  0.538469310105683, 0.000000000000000],
                    [0.932469514203152, 0.661209386466265, 0.238619186083197],
                    [0.949107912342759, 0.741531185599394, 0.405845151377397,
                     0.000000000000000],
                    [0.960289856497536, 0.796666477413627,  0.525532409916329,
                     0.183434642495650],
                    [0.968160239507626, 0.836031107326636, 0.613371432700590,
                     0.324253423403809, 0.000000000000000],
                    [0.973906528517172, 0.865063366688985, 0.679409568299024,
                     0.433395394129247, 0.148874338981631],
                    [0.978228658146057, 0.887062599768095, 0.730152005574049,
                     0.519096129110681, 0.269543155952345, 0.000000000000000],
                    [0.981560634246719, 0.904117256370475, 0.769902674194305,
                     0.587317954286617, 0.367831498918180, 0.125333408511469],
                    [0.984183054718588, 0.917598399222978, 0.801578090733310,
                     0.642349339440340, 0.448492751036447, 0.230458315955135,
                     0.000000000000000],
                    [0.986283808696812, 0.928434883663574, 0.827201315069765,
                     0.687292904811685, 0.515248636358154, 0.319112368927890,
                     0.108054948707344],
                    [0.987992518020485, 0.937273392400706, 0.8482065834104270,
                     0.724417731360170, 0.570972172608539, 0.394151347077563,
                     0.201194093997435, 0.000000000000000],
                    [0.989400934991650, 0.944575023073233, 0.865631202387832,
                     0.755404408355003, 0.617876244402644, 0.458016777657227,
                     0.281603550779259, 0.095012509837637]]

        # Big list of the coefficients up to 16.
        big_coef = [[2.0],
                   [1.000000000000000],
                   [0.555555555555556, 0.888888888888889],
                   [0.347854845137454, 0.652145154862546],
                   [0.236926885056189, 0.478628670499366, 0.568888888888889],
                   [0.171324492379170, 0.360761573048139, 0.467913934572691],
                   [0.129484966168870, 0.279705391489277, 0.381830050505119,
                    0.417959183673469],
                   [0.101228536290376, 0.222381034453374, 0.313706645877887,
                    0.362683783378362],
                   [0.081274388361574, 0.180648160694857, 0.260610696402935,
                    0.312347077040003, 0.330239355001260],
                   [0.066671344308688, 0.149451349150581, 0.219086362515982,
                    0.269266719309996, 0.295524224714753],
                   [0.055668567116174, 0.125580369464905, 0.186290210927734,
                    0.233193764591990, 0.262804544510247, 0.272925086777901],
                   [0.047175336386512, 0.106939325995318, 0.160078328543346,
                    0.203167426723066, 0.233492536538355, 0.249147045813403],
                   [0.040484004765316, 0.092121499837728, 0.138873510219787,
                    0.178145980761946, 0.207816047536889, 0.226283180262897,
                    0.232551553230874],
                   [0.035119460331752, 0.080158087159760, 0.121518570687903,
                    0.157203167158194, 0.185538397477938, 0.205198463721296,
                    0.215263853463158],
                   [0.030753241996117, 0.070366047488108, 0.107159220467172,
                    0.139570677926154, 0.166269205816994, 0.186161000015562,
                    0.198431485327111, 0.202578241925561],
                   [0.027152459411754, 0.062253523938648, 0.095158511682493,
                    0.124628971255534, 0.149595988816577, 0.169156519395003,
                    0.182603415044924, 0.189450610455069]]

        # Get the needed roots and coefficients for the quadrature
        #   method from the big lists and store them in self.
        roots = np.array(big_roots[n-1])
        coef = np.array(big_coef[n-1])
        if (n % 2 == 1):
            eval_pts = np.concatenate([-roots, roots[::-1][1:]])
            self.weights = np.concatenate([coef, coef[::-1][1:]])
        else:
            eval_pts = np.concatenate([-roots, roots[::-1]])
            self.weights = np.concatenate([coef, coef[::-1]])
        self.evaluation_pts = term1*eval_pts + term2

    def shade_under_curve(self, fig, f):
        """
        shade_under_curve draws a plot of f from a to b and adds a
        lagrange fit that shows a pictorial representation of
        Gaussian-Legendre Quadrature.

        Parameters
        ----------
        fig : The figure in which to add the plots
        f : The function to be plotted.
        """
        # Add a subplot to the supplied figure.
        ax = fig.add_subplot(111)

        # Create the Lagrange Interpolation fit.
        x_points = self.evaluation_pts
        y_points = f(x_points)
        poly_fit = scibary(x_points, y_points)

        # Create the points to be graphed.
        xL = np.linspace(self.a, self.b, 150)
        yL = poly_fit(xL)

        # Draw yellow lines going from the x-axis to the evaluation
        # points.
        for x, y in zip(x_points, y_points):
            ax.plot([x, x], [0, y], 'y-')

        # Plot the Lagrange fit and shade underneath.
        ax.plot(xL, yL, 'y-')
        ax.fill_between(xL, yL, color='y', alpha='0.25')

        # Plot the actual function and set appropriate x-axis limits.
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])


class MonteCarlo:
    """
    This class defines the numerical integration method Monte-Carlo
    Integration.

    It works by creating n test points to approximate the ratio of the
    area under the curve and the functions bounding box in the
    interval.

    MonteCarlo does not inherit the Quadrature class because it's not a
    quadrature method. Also, it cannot approximate any integral
    perfectly.
    >>> leftLimit = 0
    >>> rightLimit = 2*np.pi
    >>> numDarts = 5000
    >>> Q = MonteCarlo(leftLimit, rightLimit, numDarts)
    >>> f = lambda x: np.sin(x)
    >>> I = Q.integrate(f)
    >>> I < 1e-1 and I > -1e-1
    True
    >>> F = lambda x: -np.cos(x)
    >>> E = Q.error(F)
    >>> abs(E) < 1e-1
    True
    """
    def __init__(self, a, b, n):
        """
        Initializes the left-endpoint: a, the right-endpoint: b, and
        the number of evaluations: n.

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of evaluations.
        """
        self.a = a
        self.b = b
        self.n = n

    def integrate(self, f):
        """
        Integrates the function, f, by means of Monte-Carlo
        Integration. That is, it is computed by finding the ratio of
        the area taken up by the function and multiplying it by the
        area of its bounding box.

        Parameters
        ----------
        f : a function of one variable

        Returns
        -------
        approx : The approximate integral.
        """
        a, b, n = self.a, self.b, self.n

        # Start timing the approximation.
        start = time.time()

        # Create an array of the actual y values to be compared to,
        #   and find the min and max of the function over [a,b].
        self.x_pts = np.linspace(a, b, n)
        x_pts = self.x_pts
        y_pts = f(x_pts)
        max_value = np.amax(y_pts)
        min_value = np.amin(y_pts)

        # If the function is always positive, just check if the random
        #   values are under the curve.
        if min_value >= 0:
            # Make an array of n random numbers from 0 to the max value
            #   of the function over [a,b].
            self.rand = np.random.rand(n) * max_value
            rand = self.rand

            # Approximate the ratio of the area under the curve and the
            #   bounding box of the function.
            ratio = np.sum(rand <= y_pts) / float(n)

            # Approximate the area under the curve.
            self.approx = ratio * max_value * (b - a)
        # If the function is always negative, just check if the random
        #   values are above the curve.
        elif max_value <= 0:
            # Make an array of n random numbers from the min-value of
            #   the function over [a,b] to 0.
            self.rand = np.random.rand(n) * min_value
            rand = self.rand

            # Approximate the ratio of the area over the curve and the
            #   bounding box of the function.
            ratio = np.sum(rand >= y_pts) / float(n)

            # Approximate the area over the curve. Note that
            #   multiplying by min_value assures a negative number.
            self.approx = ratio * min_value * (b - a)
        # If the function is both negative and positive at different
        #   points, make different Monte Carlos.
        else:
            # Make an array of n random numbers.
            bigRand = np.random.rand(n)

            # Find an approximate area for the sections above the x-axis.
            yPtsOverZero = y_pts >= 0
            sumOverZero = np.sum(yPtsOverZero)
            randOverZero = np.zeros(n)
            randOverZero[yPtsOverZero] = bigRand[yPtsOverZero]*max_value
            ratioOverZero = float(np.sum(randOverZero[yPtsOverZero] <=
                                  y_pts[yPtsOverZero]))/sumOverZero
            areaOverZero = ratioOverZero*(b - a)*sumOverZero/n*max_value

            # Find an approximate area for the sections below the x-axis.
            yPtsUnderZero = y_pts < 0
            sumUnderZero = np.sum(yPtsUnderZero)
            randUnderZero = np.zeros(n)
            randUnderZero[yPtsUnderZero] = bigRand[yPtsUnderZero]*min_value
            ratioUnderZero = float(np.sum(randUnderZero[yPtsUnderZero] >=
                                          y_pts[yPtsUnderZero]))/sumUnderZero
            areaUnderZero = ratioUnderZero*(b - a)*sumUnderZero/n*min_value

            # Stores the approximated area under the curve and the random
            #     values used.
            self.rand = randOverZero + randUnderZero
            self.approx = areaOverZero + areaUnderZero

        # Finish timing the approximation.
        end = time.time()
        self.time_taken = end - start
        return self.approx

    def error(self, F):
        """
        Figures out the absolute error of the approximated integral.

        Parameters
        ----------
        F : a function of one variable that is the antiderivative of
            the function that is being approximated.

        Returns
        -------
        The absolute error.

        Notes
        -----
        This method must be called AFTER integrate(f).
        """
        return abs((F(self.b) - F(self.a)) - self.approx)

    def shade_under_curve(self, fig, f):
        """
        shade_under_curve draws a plot of f from a to b and adds all
        the evaluations used to approximate the ratio of the area under
        the curve and area of the function's bounding box.

        Parameters
        ----------
        fig : The figure in which to add the plots
        f : The function to be plotted.
        """
        # Add a subplot to the supplied figure.
        ax = fig.add_subplot(111)

        # Get the x and y points used to approximate the area.
        x_points = self.x_pts
        y_points = self.rand

        # Plot the dots and the actual curve.
        ax.plot(x_points, y_points, 'yo')
        xL = np.linspace(self.a, self.b, 150)
        yL = f(xL)
        ax.plot(xL, yL, 'b-')
        ax.set_xlim([self.a, self.b])

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    f = lambda x: 2 * np.sqrt(1 - x**2)
    Q = Simpsons(-1, 1, 4)
    I = Q.integrate(f)
    F = lambda x: np.pi * x/2.0
    fig1 = plt.figure()
    Q.shade_under_curve(fig1, f)
    print(Q.error(F))
    print(I)
    print(Q.time_taken)
    plt.show()
