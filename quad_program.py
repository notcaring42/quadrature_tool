import src.quadrature as quad
import sys

from numpy import *
from PyQt4 import QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg \
    import FigureCanvasQTAgg as FigureCanvas
from gui.main_window import Ui_main_window
from src.utils import MethodNames, parse_function


class QuadMethod:
    """Holds a Quadrature object, references to the inputs
    used to change it, and the name of the object's class,
    and is responsible for updating the object.

    Attributes
    ----------
    quadrature: the Quadrature object
    method: a reference to the class constructor, used
        to create a new value for quadrature
    lx_input: QLineEdit used to define the lower-x bound (a)
    rx_input: QLineEdit used to define the upper-x bound (b)
    n_input: QLineEdit used to define n, the number of polygons
        to use in quadrature
    def_n: the default n value that the Quadrature object should use
    max_n: the maximum n value that the Quadrature object can use
    needs_even: boolean representing whether the Quadrature object
        can only use even values of n

    Notes
    -----
    The update function is actually called by QuadUpdater, which holds
    references to numerous QuadMethods
    """
    def __init__(self, top_widget, class_name, n_input,
                 def_n=20, max_n=1500, needs_even=False):
        self.lx_input = top_widget.findChild((QtGui.QLineEdit,),
                                             'x_Left_Line_Edit')
        self.rx_input = top_widget.findChild((QtGui.QLineEdit,),
                                             'x_Right_Line_Edit')
        self.n_input = n_input
        self.class_name = class_name
        self.def_n = def_n
        self.max_n = max_n
        self.needs_even = needs_even
        self.method = getattr(quad, class_name)
        self.update()

    def update(self):
        # Parse lx
        try:
            lx = float(self.lx_input.text())
        except ValueError:
            lx = -7.0
            self.lx_input.setText('-7.0')
        # Parse rx
        try:
            rx = float(self.rx_input.text())
        except ValueError:
            rx = 7.0
            self.rx_input.setText('7.0')

        # Parse n
        try:
            n = int(self.n_input.text())
        except ValueError:
            n = self.def_n
            self.n_input.setText(str(self.def_n))

        # Validate n
        if n > self.max_n:
            n = self.max_n
            self.n_input.setText(str(self.max_n))

        if self.needs_even and n % 2 != 0:
            n = self.def_n
            self.n_input.setText(str(self.def_n))

        self.quadrature = self.method(lx, rx, n)


class QuadUpdater:
    def __init__(self, top_widget):
        self.quad_methods = self.create_dictionary(top_widget)

        update_button = top_widget.findChild((QtGui.QPushButton,),
                                             'update_button')
        update_button.clicked.connect(self.update)

        self.update()

    def update(self):
        for quad_method in self.quad_methods.values():
            quad_method.update()

    def create_dictionary(self, top_widget):
        kvp = []

        kvp.append(('lpr', QuadMethod(top_widget, 'LeftPointRiemman',
                   top_widget.findChild((QtGui.QLineEdit,), 'rl_n'))))

        kvp.append(('rpr', QuadMethod(top_widget, 'RightPointRiemman',
                   top_widget.findChild((QtGui.QLineEdit,), 'rr_n'))))

        kvp.append(('mpr', QuadMethod(top_widget, 'MidPointRiemman',
                   top_widget.findChild((QtGui.QLineEdit,), 'rm_n'))))

        kvp.append(('trap', QuadMethod(top_widget, 'Trapezoidal',
                   top_widget.findChild((QtGui.QLineEdit,), 'trap_n'))))

        kvp.append(('simp', QuadMethod(top_widget, 'Simpsons',
                   top_widget.findChild((QtGui.QLineEdit,), 'simp_n'),
                   needs_even=True)))

        kvp.append(('gauss', QuadMethod(top_widget, 'Gaussian',
                   top_widget.findChild((QtGui.QLineEdit,), 'gauss_n'),
                   def_n=5, max_n=16)))

        kvp.append(('monte', QuadMethod(top_widget, 'MonteCarlo',
                   top_widget.findChild((QtGui.QLineEdit,), 'monte_n'),
                   def_n=500)))

        return dict(kvp)


class FunctionUpdater:
    def __init__(self, top_widget):
        self.f_input = top_widget.findChild((QtGui.QLineEdit,),
                                            'f_input')
        self.F_input = top_widget.findChild((QtGui.QLineEdit,),
                                            'F_input')

        update_button = top_widget.findChild((QtGui.QPushButton,),
                                             'update_button')
        update_button.clicked.connect(self.update)
        self.update()

    def update(self):
        global f_str
        f_str = parse_function(str(self.f_input.text()))

        exec """def f(x):
            return eval(f_str)"""

        global F_str
        F_str = parse_function(str(self.F_input.text()))

        exec """def F(x):
            return eval(F_str)"""

        self.f = f
        self.F = F


class MplCanvas(FigureCanvas):
    def __init__(self, top_widget, quad_methods, function_updater,
                 parent=None, width=5, height=4, dpi=100):
        self.quad_methods = quad_methods
        self.function_updater = function_updater
        self.method_group = top_widget.findChild((QtGui.QButtonGroup,),
                                                 'method_group')
        self.result = top_widget.findChild((QtGui.QLabel,),
                                           'calculated_result')
        self.error = top_widget.findChild((QtGui.QLabel,),
                                          'error_calculated')
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        update_button = top_widget.findChild((QtGui.QPushButton,),
                                             'update_button')
        update_button.clicked.connect(self.update_figure)

        self.update_figure()

    def update_figure(self):
        for axis in self.fig.axes:
            axis.cla()

        m_i = -2 - self.method_group.checkedId()

        if m_i == MethodNames.LEFT_POINT_RIEMMAN:
            Q = self.quad_methods['lpr'].quadrature
        elif m_i == MethodNames.RIGHT_POINT_RIEMMAN:
            Q = self.quad_methods['rpr'].quadrature
        elif m_i == MethodNames.MID_POINT_RIEMMAN:
            Q = self.quad_methods['mpr'].quadrature
        elif m_i == MethodNames.TRAPEZOIDAL:
            Q = self.quad_methods['trap'].quadrature
        elif m_i == MethodNames.SIMPSONS:
            Q = self.quad_methods['simp'].quadrature
        elif m_i == MethodNames.GAUSSIAN:
            Q = self.quad_methods['gauss'].quadrature
        elif m_i == MethodNames.MONTE_CARLO:
            Q = self.quad_methods['monte'].quadrature
        else:
            raise IndexError('checkedId did not correspond ' +
                             'to a valid method')

        try:
            self.result.setText(str(Q.integrate(self.function_updater.f)))
            Q.shadeUnderCurve(self.fig, self.function_updater.f)
            try:
                self.error.setText(str(Q.error(self.function_updater.F)))
            except SyntaxError:
                self.error.setText('N/A')
        except SyntaxError:
            self.result.setText('N/A')
            self.error.setText('N/A')
            self.fig.clf()
        self.draw()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Ui_main_window()
    widget = QtGui.QWidget()
    window.setupUi(widget)
    scroll = QtGui.QScrollArea()
    scroll.setWidget(widget)
    qu = QuadUpdater(widget)
    graph = MplCanvas(widget, qu.quad_methods, FunctionUpdater(widget))
    graph_layout = widget.findChild((QtGui.QVBoxLayout,),
                                    'graph_layout')
    graph_layout.addWidget(graph)
    scroll.resize(700, 800)
    scroll.setWindowTitle('Numerical Integrator')
    scroll.show()
    sys.exit(app.exec_())
