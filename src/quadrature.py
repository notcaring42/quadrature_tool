import numpy as np
from scipy.interpolate import BarycentricInterpolator as scibary
import matplotlib.pyplot as plt
import time


class Quadrature:
    def __init__(self, a, b, n):
        """\
        Initialize the left-endpoint: a, the right endpoint: b, and the number
        of intervals: n. Also creates the length of the interval: h.
    

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        self.a = a
        self.b = b
        self.n = n
        self.h = float((b-a))/n

    def integrate(self, f):
        """\
        Integrates the function, f, by means of a quadrature method. That is,
        it is computed by sum(c*f(x)).
    

        Parameters
        ----------
        f : a function of one variable

        Returns
        -------
        approx : The approximate integral.

        Notes
        -----
        If all the weights are one, it is best to instead set them equal to
        None and this method will skip the wasted multiplication.
        """
        start = time.time()
        try:
            if (self.weights == None):
                self.approx = self.outer*sum(f(self.evaluationPts))
            else:
                self.approx = self.outer*sum(self.weights*f(self.evaluationPts))
        except SyntaxError:
            return 'N/A'
            end = time.time()
            self.timeTaken = end - start
        end = time.time()
        self.timeTaken = end - start
        return self.approx

    
    def error(self, F):
        """\
        Figures out the absolute error of the approximated integral.
    

        Parameters
        ----------
        F : a function of one variable that is the antiderivative of the
            function that is being approximated.

        Returns
        -------
        The absolute error.

        Notes
        -----
        This method must be called AFTER integrate(f). If not, an exception
        will be thrown. I didn't implement the exception yet though.
        """
        return abs( (F(self.b) - F(self.a)) - self.approx )


class LeftPointRiemman(Quadrature):
    def __init__(self,a, b, n):
        """\
        Initialize the left-endpoint: a, the right-endpoint: b, and the number
        of intervals: n. Left-Point Riemman summations follow the quadrature
        formula h*sum(f(x_i)) where x_i = a + i*h from i = [0..n-1].  
    

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self,a, b, n)
        self.evaluationPts = np.linspace(a, b-self.h, n)
        self.weights = None
        self.outer = self.h

    def shadeUnderCurve(self, fig, f):
        """\
        shadeUnderCurve adds shaded in rectangles evaluationed at the left 
        end-point to a figure and draws a plot of the function: f.
    

        Parameters
        ----------
        fig : The figure in which to add the rectangles.
        f : The function to be plotted.
        """
        # Adds a subplot to the supplied figure.
        ax = fig.add_subplot(111)
        
        # Gets the points to make the rectangles.
        xPoints = np.linspace(self.a, self.b, self.n+1)
        xPoints1 = xPoints[:-1]
        xPoints2 = xPoints[1:]

        yPoints = f(self.evaluationPts)

        # Plots the rectangles and shades them in.
        for x1, x2, y in zip(xPoints1, xPoints2, yPoints):
            ax.plot([x1, x1], [0, y], 'y-')
            ax.plot([x2, x2], [0, y], 'y-')
            ax.plot([x1, x2], [y, y], 'y-')
            ax.fill_between([x1, x2], [y, y], color = 'y', alpha = 0.25)    

        # Plots the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])




class RightPointRiemman(Quadrature):
    def __init__(self,a, b, n):
        """\
        Initialize the left-endpoint: a, the right-endpoint: b, and the number
        of intervals: n. Right-Point Riemman summations follow the quadrature
        formula h*sum(f(x_i)) where x_i = a + i*h from i = [1..n].  
    

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self,a, b, n)
        self.evaluationPts = np.array(np.linspace(a + self.h, b, n))
        self.weights = None
        self.outer = self.h

    
    def shadeUnderCurve(self, fig, f):
        """\
        shadeUnderCurve adds shaded in rectangles evaluationed at the right 
        end-point to a figure and draws a plot of the function: f.
    

        Parameters
        ----------
        fig : The figure in which to add the rectangles.
        f : The function to be plotted.
        """
        # Adds a subplot to the supplied figure.
        ax = fig.add_subplot(111)
        
        # Gets the points to make the rectangles.
        xPoints = np.linspace(self.a, self.b, self.n+1)
        xPoints1 = xPoints[:-1]
        xPoints2 = xPoints[1:]
        yPoints = f(self.evaluationPts)

        # Plots the rectangles and shades them in.
        for x1, x2, y in zip(xPoints1, xPoints2, yPoints):
            ax.plot([x1, x1], [0, y], 'y-')
            ax.plot([x2, x2], [0, y], 'y-')
            ax.plot([x1, x2], [y, y], 'y-')
            ax.fill_between([x1, x2], [y, y], color = 'y', alpha = 0.25)    

        # Plots the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])




class MidPointRiemman(Quadrature):
    def __init__(self,a, b, n):
        """\
        Initialize the left-endpoint: a, the right endpoint: b, and the number
        of intervals: n. Mid-Point Riemman Summation follow the quadrature
        formula h*sum(f(x_i)) where x_i = a + h/2 + i*h from i = [0..n-1].  
    

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self,a, b, n)
        self.evaluationPts = np.array(np.linspace(a+self.h/2.0, b-self.h/2.0, n))
        self.weights = None
        self.outer = self.h

    def shadeUnderCurve(self, fig, f):
        """\
        shadeUnderCurve adds shaded in rectangles evaluationed at the mid point
        to a figure and draws a plot of the function: f.
    

        Parameters
        ----------
        fig : The figure in which to add the trapezoids or rectangles.
        f : The function to be plotted.
        """
        # Adds a subplot to the supplied figure.
        ax = fig.add_subplot(111)
        
        # Gets the points to make the rectangles.
        xPoints = np.linspace(self.a, self.b, self.n+1)
        xPoints1 = xPoints[:-1]
        xPoints2 = xPoints[1:]
        yPoints = f(self.evaluationPts)

        # Plots the rectangles and shades them in.
        for x1, x2, y in zip(xPoints1, xPoints2, yPoints):
            ax.plot([x1, x1], [0, y], 'y-')
            ax.plot([x2, x2], [0, y], 'y-')
            ax.plot([x1, x2], [y, y], 'y-')
            ax.fill_between([x1, x2], [y, y], color = 'y', alpha = 0.25)    

        # Plots the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])


class Trapezoidal(Quadrature):
    def __init__(self,a, b, n):
        """\
        Initialize the left-endpoint: a, the right endpoint: b, and the number
        of intervals: n. The Composite Trapezoidal Rule follows the quadrature
        formula h/2*(f(a) + sum(2*f(x_i)) + f(b)) where x_i = a + i*h from 
        i = [1..n-1].  
    

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self,a, b, n)
        self.evaluationPts = np.linspace(a, b, n+1)
        self.weights = np.concatenate([[1], np.zeros(n - 1) + 2, [1]])
        self.outer = self.h/2.0

    def shadeUnderCurve(self, fig, f):
        """\
        shadeUnderCurve adds Shaded in Trapezoids to a figure from adjacent
        evaluation points and draws a plot of the function: f.
    

        Parameters
        ----------
        fig : The figure in which to add the trapezoids or rectangles.
        f : The function to be plotted.
        """
        # Adds a subplot to the supplied figure.
        ax = fig.add_subplot(111)
        
        # Gets the points to make the trapezoids.
        xPoints = self.evaluationPts
        yPoints = f(xPoints)
        plotPts = zip(xPoints, yPoints)
        pts1 = plotPts[:-1]
        pts2 = plotPts[1:]

        # Plots the trapezoids and shades them in.
        for pt1, pt2 in zip(pts1, pts2):
            x1, y1 = pt1[0], pt1[1]
            x2, y2 = pt2[0], pt2[1]
            ax.plot([x1, x1], [0, y1], 'y-')
            ax.plot([x2, x2], [0, y2], 'y-')
            ax.plot([x1, x2], [y1, y2], 'y-')
            ax.fill_between([x1, x2], [y1, y2], color = 'y', alpha = 0.25)  

        # Plots the actual function.
        xL = np.linspace(self.a, self.b, 150)
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])


class Simpsons(Quadrature):
    def __init__(self, a, b, n):
        """\
        Initialize the left-endpoint: a, the right endpoint: b, and the number
        of intervals: n. The Composite Simpsons Rule follows the quadrature
        formula h/3*(f(a) + sum(4*f(x_i)) + sum(2*f(x_j) + f(b)) where 
        x_i = a + (2*i-1)*h from i = [1..floor(n/2)] and x_j = a + 2*j*h 
        from j = [1..floor(n/2)-1].
    

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self,a, b, n)
        self.evaluationPts = np.linspace(a, b, n+1)
        self.weights = np.concatenate( (np.array([1]), \
             np.arange(2, 2*n, 2)%4 + 2, np.array([1])) )
        self.outer = self.h/3.0

    def shadeUnderCurve(self, fig, f):
        """\
        Adds floor(n/2) quadratic lagrange interpolants using 3 adjacent points
        to fig and shades them underneath. Also draws a plot of f.

        Parameters
        ----------
        fig : The figure in which to add the trapezoids or rectangles.
        f : The function to be plotted.
        """
        # Adds a sub plot to the supplied figure.
        ax = fig.add_subplot(111)

        # Creates the points to make the Quadratic fits.
        xPoints = self.evaluationPts
        yPoints = f(xPoints)
        points = zip(xPoints,yPoints)   
        pts1 = points[0:-1:2]
        pts2 = points[1::2]
        pts3 = points[2::2]
        plotPts = [(pt1,pt2,pt3) for pt1,pt2,pt3 in zip(pts1, pts2, pts3)]
        
        # plots the Quadratic fits and shades them in.
        for pts in plotPts:
            xPts = [p[0] for p in pts]
            yPts = [p[1] for p in pts]
            PolyFit = scibary(xPts, yPts)
            x1 = xPts[0]
            x2 = xPts[-1]
            xL = np.linspace(x1, x2, 300/self.n)
            yL = PolyFit(xL)
            for x,y in pts:
                ax.plot([x, x], [0, y], 'y-')
            ax.plot(xL, yL, 'y-')
            ax.fill_between(xL, yL, color = 'y', alpha = '0.25')
        
        # Plots the actual function.
        xL2 = np.linspace(self.a, self.b , 150)
        ax.plot(xL2, f(xL2))
        ax.set_xlim([self.a,self.b])

class Gaussian(Quadrature): 
    def __init__(self, a, b, n):
        """\
        Initialize the left-endpoint: a, the right endpoint: b, and the number
        of intervals: n. The Gaussian-Legendre method follows the quadrature
        formula (b - a)/2*sum(c_i*f((b - a)/2*x_i + (b + a)/2)) where x_i are
        the roots of the nth legendre polynomials and c_i is the associated
        lagrange interpolant weight. 
    

        Parameters
        ----------
        a : The left endpoint of the integral.
        b : The right endpoint of the integral.
        n : The number of intervals between function evaluations.
        """
        Quadrature.__init__(self, a, b, n)

        # Creates the terms for transforming the interval from [-1,1] -> [a,b]
        term1 = (b-a)/2.0
        term2 = (b+a)/2.0
        self.outer = term1 

        # Big list of the roots of the legendre polynomials up to 16.       
        BigRoots = [[0.0],
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
        BigCoef =  [[2.0],
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






        
        # Gets the needed roots and coefficients for the quadrature method 
        #   from the big list and stores them in self.
        roots = np.array(BigRoots[n-1])
        coef = np.array(BigCoef[n-1])
        if (n%2 == 1):
            evalPts = np.concatenate([-roots, roots[::-1][1:]])
            self.weights = np.concatenate([coef, coef[::-1][1:]])
        else:
            evalPts = np.concatenate([-roots, roots[::-1]])
            self.weights = np.concatenate([coef, coef[::-1]])
        self.evaluationPts = term1*evalPts + term2

    def shadeUnderCurve(self, fig, f):
        """\
        shadeUnderCurve draws a plot of f from a to b and adds a lagrange
        fit that shows a pictorial representation of Gaussian-Legendre
        Quadrature.
        
        Parameters
        ----------
        fig : The figure in which to add the plots
        f : The function to be plotted.
        """
        # Adds a subplot to the supplied figure.
        ax = fig.add_subplot(111) 

        # Create the Lagrange Interpolation fit.
        xPoints = self.evaluationPts 
        yPoints = f(xPoints)
        PolyFit = scibary(xPoints, yPoints)     

        # Create the points to be graphed.
        xL = np.linspace(self.a, self.b , 150)
        yL = PolyFit(xL)

        # Draw yellow lines going from the x-axis to the evaluation points.
        for x,y in zip(xPoints, yPoints):
            ax.plot([x, x], [0, y], 'y-')

        # Plot the Lagrange fit and shade underneath.
        ax.plot(xL, yL, 'y-')
        ax.fill_between(xL, yL, color = 'y', alpha = '0.25')

        # Plot the actual function and set appropriate x-axis limits.
        ax.plot(xL, f(xL))
        ax.set_xlim([self.a, self.b])
        
        



class MonteCarlo:
    def __init__(self, a, b, n):
        """\
        Initializes the left-endpoint: a, the right-endpoint: b, and the number
        of evaluations: n.
        
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
        """\
        Integrates the function, f, by means of Monte-Carlo Integration. That
        is, it is computed by finding the ratio of the area taken up by the
        function and multiplying it by the area of its bounding box.
    

        Parameters
        ----------
        f : a function of one variable

        Returns
        -------
        approx : The approximate integral.
        """
        a, b, n = self.a, self.b, self.n

        start = time.time()
        # Creates an array of the actual y values to be compared to, and the
        #   min and max of the function over [a,b].
        self.xPts = np.linspace(a, b, n)
        xPts = self.xPts
        yPts = f(xPts)
        maxValue = max(yPts)
        minValue = min(yPts)

        # If the function is always positive.
        if minValue >= 0:
            self.rand = np.random.rand(n)*maxValue
            rand = self.rand
            ratio = sum(rand <= yPts)/float(n)
            self.approx = ratio*maxValue*(b - a)
        # If the function is always negative.
        elif maxValue <= 0:
            self.rand = np.random.rand(n)*minValue
            rand = self.rand            
            ratio = sum(rand >= yPts)/float(n)
            self.approx = ratio*minValue*(b - a)
        # If the function is both negative and positive at different points.
        else:
            # Makes an array of n random numbers.
            bigRand = np.random.rand(n)
            
            # Find an approximate area for the sections above the x-axis.
            yPtsOverZero = yPts >= 0
            sumOverZero = sum(yPtsOverZero)
            randOverZero = np.zeros(n)
            randOverZero[yPtsOverZero] = bigRand[yPtsOverZero]*maxValue
            ratioOverZero = float(sum(randOverZero[yPtsOverZero] <= yPts[yPtsOverZero]))/sumOverZero
            areaOverZero = ratioOverZero*(b - a)*sumOverZero/n*maxValue 

            # Find an approximate area for the sections below the x-axis.
            yPtsUnderZero = yPts < 0
            sumUnderZero = sum(yPtsUnderZero)
            randUnderZero = np.zeros(n)
            randUnderZero[yPtsUnderZero] = bigRand[yPtsUnderZero]*minValue
            ratioUnderZero = float(sum(randUnderZero[yPtsUnderZero] >= yPts[yPtsUnderZero]))/sumUnderZero
            areaUnderZero = ratioUnderZero*(b - a)*sumUnderZero/n*minValue
                
            # Stores the approximated area under the curve and the random 
            #   values used.
            self.rand = randOverZero + randUnderZero
            self.approx = areaOverZero + areaUnderZero
        end = time.time()
        self.timeTaken = end - start
        return self.approx

    def error(self, F):
        """\
        Figures out the absolute error of the approximated integral.
    

        Parameters
        ----------
        F : a function of one variable that is the antiderivative of the
            function that is being approximated.

        Returns
        -------
        The absolute error.

        Notes
        -----
        This method must be called AFTER integrate(f). If not, an exception
        will be thrown. I didn't implement the exception yet though.
        """
        return abs( (F(self.b) - F(self.a)) - self.approx )

    def shadeUnderCurve(self, fig, f):
        """\
        shadeUnderCurve draws a plot of f from a to b and adds all the
        evaluations used to approximate the ratio of the area under the curve
        and area of the function's bounding box.
        
        Parameters
        ----------
        fig : The figure in which to add the plots
        f : The function to be plotted.
        """
        # Adds a subplot to the supplied figure.
        ax = fig.add_subplot(111)
            
        # Gets the x and y points used to approximate the area.
        xPoints = self.xPts
        yPoints = self.rand

        # Gets the x and y values to draw the actual plot.
        xL = np.linspace(self.a, self.b, 150)
        yL = f(xL)
        
        # Plots the dots and the actual plot.
        for x,y in zip(xPoints, yPoints):
            ax.plot(x,y,'yo')
        ax.plot(xL, yL, 'b-')
    

if __name__ == '__main__':
    from math import pi
    Q = Simpsons(-2,1,2)
    def f(x):
        return x**3
    def F(x):
        return x**4/4.0

    fig2 = plt.figure()
    print(Q.integrate(f))
    print(Q.timeTaken)
    print(Q.error(F))

    Q.shadeUnderCurve(fig2, f)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()



