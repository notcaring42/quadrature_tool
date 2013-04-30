import sys

from numpy import *
from PyQt4 import QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg \
    import FigureCanvasQTAgg as FigureCanvas

import src.quadrature as quad
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
                 def_n=20, max_n=3000, needs_even=False):
        """Creates a new QuadMethod

        Parameters
        ----------
        top_widget: the top level widget of the application
        class_name: the name of the quadrature method's class
        n_input: QLineEdit used to input n for this quadrature method
        def_n: the default value for n that the quadrature method
            should use
        max_n: the maximum value for n that the quadrature method
            can use
        needs_even: boolean representing whether the Quadrature object
            can only use even values of n

        Returns
        -------
        a new QuadMethod
        """
        self.lx_input = top_widget.findChild((QtGui.QLineEdit,),
                                             'x_Left_Line_Edit')
        self.rx_input = top_widget.findChild((QtGui.QLineEdit,),
                                             'x_Right_Line_Edit')
        self.function_updater = FunctionUpdater(top_widget)
        self.n_input = n_input
        self.class_name = class_name
        self.def_n = def_n
        self.max_n = max_n
        self.needs_even = needs_even

        # Grab the class and constructor for this
        # QuadMethod from quadrature.py
        self.method = getattr(quad, class_name)

        # Call update for the first time to initialize values
        self.update()

    def update(self):
        """Updates the Quadrature object for this QuadMethod."""
        # Parse lx
        try:
            lx = float(self.lx_input.text())
        except ValueError:
            # If the input is invalid, set the default rx value
            lx = -7.0
            self.lx_input.setText('-7.0')

        # Parse rx
        try:
            rx = float(self.rx_input.text())
        except ValueError:
            # If the input is invalid, set the default rx value
            rx = 7.0
            self.rx_input.setText('7.0')

        # Parse n
        try:
            n = int(self.n_input.text())
        except ValueError:
            # If the input is valid, set the default n value
            n = self.def_n
            self.n_input.setText(str(self.def_n))

        # Certain methods will need special validation for n
        # If n > the max value it can handle, we set the value
        # to the method's maximum value
        if n > self.max_n:
            n = self.max_n
            self.n_input.setText(str(self.max_n))

        # If the method needs an even n and if n is odd,
        # set the default n
        if self.needs_even and n % 2 == 1:
            n = n - 1
            self.n_input.setText(str(n))

        # Finally, we create the Quadrature object
        # using the parsed values and the reference to the
        # class constructor
        self.quadrature = self.method(lx, rx, n)


class QuadUpdater:
    """Holds a dictionary of QuadMethods and is responsible
    for updating them

    Attributes
    ----------
    quad_methods: a dictionary of QuadMethods
    quad_check_buttons: a list of tuples matching method names
        and matching checkboxes
    active_quad_method: the current quadrature method being graphed
    result: QLabel used to display the result of the integration
    error: QLabel used to display the error of the integration
    method_group: QButtonGroup used to select the currently
        active quadrature method
    function_updater: FunctionUpdater used to grab f(x)
        and F(x)

    Notes
    -----
    This class is configured to update the QuadMethods when the
    'Update' button is clicked
    """
    def __init__(self, top_widget):
        """Creates a new QuadUpdater

        Parameters
        ----------
        top_widget: the top level widget of the application

        Returns
        -------
        a new QuadUpdater, configured to update on the press
        of the 'Update button'
        """
        # Create the QuadMethod dictionary
        self.quad_methods = self.create_method_dict(top_widget)
        self.quad_check_buttons = self.create_button_list(top_widget)
        self.result = top_widget.findChild((QtGui.QLabel,),
                                           'calculated_result')
        self.error = top_widget.findChild((QtGui.QLabel,),
                                          'error_calculated')
        self.method_group = top_widget.findChild((QtGui.QButtonGroup,),
                                                 'method_group')
        self.quad_table = top_widget.findChild((QtGui.QTableWidget,),
                                               'quad_method_table')
        self.function_updater = FunctionUpdater(top_widget)

        # Register the update function with the update button
        update_button = top_widget.findChild((QtGui.QPushButton,),
                                             'update_button')
        update_button.clicked.connect(self.update)

        # Call update for the first time to initialize the QuadMethods
        self.update()

    def update(self):
        """Updates each QuadMethod tracked by this QuadUpdater,
        determines the currently active quadrature method
        from the method_group, and displays the result/
        error of integration
        """
        for quad_method in self.quad_methods.values():
            quad_method.update()

        # Grab the checked id of the button group
        # to determine which quadrature method to graph.
        # Grab the corresponding Quadrature object from
        # the quad_methods dictionary
        m_i = -2 - self.method_group.checkedId()

        if m_i == MethodNames.LEFT_POINT_RIEMMAN:
            Q = self.quad_methods['lpr']
        elif m_i == MethodNames.RIGHT_POINT_RIEMMAN:
            Q = self.quad_methods['rpr']
        elif m_i == MethodNames.MID_POINT_RIEMMAN:
            Q = self.quad_methods['mpr']
        elif m_i == MethodNames.TRAPEZOIDAL:
            Q = self.quad_methods['trap']
        elif m_i == MethodNames.SIMPSONS:
            Q = self.quad_methods['simp']
        elif m_i == MethodNames.GAUSSIAN:
            Q = self.quad_methods['gauss']
        elif m_i == MethodNames.MONTE_CARLO:
            Q = self.quad_methods['monte']
        else:
            raise IndexError('checkedId did not correspond ' +
                             'to a valid method')

        # Set the active quadrature method
        self.active_quad_method = Q

        # Loop through each check button to determine
        # which methods to show on the table
        # i keeps track of the row number for each method
        row_number = 0
        for qm_name, qm_check in self.quad_check_buttons:
            q_m = self.quad_methods[qm_name]

            # If this method is checked for the table or if it's
            # the one we're graphing, we're going to need to grab
            # integration results
            #
            # Note: this seems a little convulted to check both
            # conditions up top and then again within the if statement,
            # but by doing this, if the method is checked and is the
            # one being graphed, we can share integration values
            # between the two rather than integrating twice. This was
            # especially noticable with Monte Carlo, where the results
            # under the graph and in the table were different (because
            # Monte Carlo is random). It also saves time!
            if qm_check.isChecked() or q_m == self.active_quad_method:
                # Display results of integration and calculated error
                # If we hit a syntax error, then we failed to
                # parse the function and we just display 'N/A'
                # Note that the error is optional because F(x) is
                # optional: therefore,the second try statement is
                # nested in the first so that we can display the result
                # (which we should always have) but not the optional
                # error
                try:
                    integrate_value = str(q_m.quadrature.
                                          integrate(self.function_updater.f))
                    time_value = q_m.quadrature.time_taken
                    time_value = '{:g}'.format(time_value)
                    try:
                        error_value = q_m.quadrature.error(self.
                                                           function_updater.F)
                        error_value = '{:g}'.format(error_value)
                    except SyntaxError:
                        error_value = ''
                except SyntaxError:
                    integrate_value = ''
                    error_value = ''
                    time_value = ''

                # If q_m is the active method, then show the result and error
                # under the graph
                if q_m == self.active_quad_method:
                    if integrate_value == '':
                        self.result.setText('N/A')
                    else:
                        self.result.setText(integrate_value)

                    if error_value == '':
                        self.error.setText('N/A')
                    else:
                        self.error.setText(error_value)

                # Get the values to display on the table. If this method
                # isn't checked, just set the values to a blank string
                if not qm_check.isChecked():
                    table_i_value = ''
                    table_e_value = ''
                    table_t_value = ''
                else:
                    table_i_value = integrate_value
                    table_e_value = error_value
                    table_t_value = time_value
            else:
                # The method isn't checked or the active method, so
                # no integration is necessary:
                # We only need to set table values, so just set the
                # values to blank strings
                table_i_value = ''
                table_e_value = ''
                table_t_value = ''

            self.quad_table.setItem(row_number, 0,
                                    QtGui.QTableWidgetItem(table_i_value))
            self.quad_table.setItem(row_number, 1,
                                    QtGui.QTableWidgetItem(table_e_value))
            self.quad_table.setItem(row_number, 2,
                                    QtGui.QTableWidgetItem(table_t_value))

            # Increment the row number
            row_number += 1

    def create_button_list(self, top_widget):
        """Creates a list of tuples matching quad methods with
        their corresponding checkboxes on the GUI.

        Parameters
        ----------
        top_widget: the top level widget of the application

        Returns
        -------
        a list of tuples matching quad methods with
        their corresponding checkboxes on the GUI.
        """
        # Create kvp tuples for each QuadMethod and its key
        # This doesn't create a real dictionary, but instead
        # a list of tuples because the order of the elements
        # will be important (dictionaries can have an order that
        # differs from how you defined it)
        kvp = []

        kvp.append(('lpr', top_widget.findChild((QtGui.QCheckBox,),
                   'rl_check')))

        kvp.append(('rpr', top_widget.findChild((QtGui.QCheckBox,),
                   'rr_check')))

        kvp.append(('mpr', top_widget.findChild((QtGui.QCheckBox,),
                   'rm_check')))

        kvp.append(('trap', top_widget.findChild((QtGui.QCheckBox,),
                   'trap_check')))

        kvp.append(('simp', top_widget.findChild((QtGui.QCheckBox,),
                   'simp_check')))

        kvp.append(('gauss', top_widget.findChild((QtGui.QCheckBox,),
                   'gauss_check')))

        kvp.append(('monte', top_widget.findChild((QtGui.QCheckBox,),
                   'monte_check')))

        # Return the list
        return kvp

    def create_method_dict(self, top_widget):
        """Creates the QuadMethod dictionary

        Parameters
        ----------
        top_widget: the top level widget of the application

        Returns
        -------
        a dictionary holding a QuadMethod for each quadrature method
        defined in quadrature.py
        """
        # Create kvp tuples for each QuadMethod and its key
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

        # Create the dictionary from the list of kvp tuples
        return dict(kvp)


class FunctionUpdater:
    """Holds references to the inputs for f(x) (the main function) and
    F(x) (the anti-derivative) and is responsible for parsing their
    inputs and updating them in accordance with the input

    Attributes
    ----------
    f: f(x), the main function to graph and integrate
    F: F(x), the anti-derivative of f(x), used for error analysis
    f_input: QLineEdit used to input f(x)
    F_input: QLineEdit used to input F(x)

    Notes
    -----
    this class is configured to update when the 'Update' button
    is pressed
    """
    def __init__(self, top_widget):
        """Creates a new FunctionUpdater

        Parameters
        ----------
        top_widget: the top level widget of the application

        Returns
        -------
        a new FunctionUpdater
        """
        self.f_input = top_widget.findChild((QtGui.QLineEdit,),
                                            'f_input')
        self.F_input = top_widget.findChild((QtGui.QLineEdit,),
                                            'F_input')

        self.tw = widget.findChild((QtGui.QTableWidget,),
                                   'tableWidget')
        # Register the update function with the update button
        update_button = top_widget.findChild((QtGui.QPushButton,),
                                             'update_button')
        update_button.clicked.connect(self.update)

        # Call update for the first time to initialize f and F
        self.update()

    def update(self):
        """Updates the functions f and F by parsing their respective
        inputs"""

        # To parse the functions, we first need to define global
        # variables to hold the strings from the QLineEdit's so that
        # the exec statement can access them.
        # We then use parse_function to turn the strings into python
        # compliant expressions.
        # Finally, we use exec to define python functions and use eval
        # to evaluate the function string as a mathematical expression.
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
    """Represents a matplotlib figure as a Qt4 widget that can
    be drawn.

    Attributes
    ----------
    fig: the matplotlib figure
    quad_updater: instance of QuadUpdater used to grab
            currently active quadrature
    function_updater: instance of FunctionUpdater used to
            graph f(x)

    Notes
    -----
    this class re-computes the figure every time the 'Update'
    button is pressed
    """
    def __init__(self, top_widget, quad_updater, function_updater,
                 parent=None, width=5, height=4, dpi=100):
        """Creates a new MplCanvas

        Parameters
        ----------
        top_widget: the top level widget of the application
        quad_updater: instance of QuadUpdater used to grab
            currently active quadrature
        function_updater: instance of FunctionUpdater used to
            graph f(x)

        Returns
        -------
        a new MplCanvas
        """
        self.quad_updater = quad_updater
        self.function_updater = function_updater

        # Create a matplotlib figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        # Initialize the parent class
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # Register the update_figure function with the update button
        update_button = top_widget.findChild((QtGui.QPushButton,),
                                             'update_button')
        update_button.clicked.connect(self.update_figure)

        # Call update_figure for the first time to initialize
        # the graph
        self.update_figure()

    def update_figure(self):
        """Updates and draws the figure"""
        # Clear out all the axes of the figure
        for axis in self.fig.axes:
            axis.cla()

        # Grab the currently active quadrature method for
        # graphing
        Q = self.quad_updater.active_quad_method.quadrature

        # Attempt to graph the function and the quadrature polygons
        # If we get a SyntaxError, then we weren't able to parse
        # a valid function: then, just clear the figure and display
        # nothing
        try:
            Q.shade_under_curve(self.fig, self.function_updater.f)
        except SyntaxError:
            self.fig.clf()

        self.draw()

# Entry point for the application
if __name__ == '__main__':
    # Initialize the application and its window
    # We add the main window to a scroll area to
    # add scroll functionality
    app = QtGui.QApplication(sys.argv)
    window = Ui_main_window()
    widget = QtGui.QWidget()
    window.setupUi(widget)
    scroll = QtGui.QScrollArea()
    scroll.setWidget(widget)
    scroll.widgetResizable = True

    # Create the canvas and add it to the window
    graph = MplCanvas(widget, QuadUpdater(widget),
                      FunctionUpdater(widget))
    graph_layout = widget.findChild((QtGui.QVBoxLayout,),
                                    'graph_layout')
    graph_layout.addWidget(graph)

    # Resize the window, set the window title, and then
    # show the window
    scroll.resize(1100, 465)
    scroll.setWindowTitle('Numerical Integrator')

    scroll.show()

    sys.exit(app.exec_())
