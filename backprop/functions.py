import math



def f1(x1, w1, x2, w2, b, y):
    """
    Computes the forward and backward pass through the computational graph f1
    from the homework PDF.

    A few clarifications about the graph:
    - The subtraction node in the graph computes d = y_hat - y
    - The ^2 node squares its input

    Inputs:
    - x1, w1, x2, w2, b, y: Python floats

    Returns a tuple of:
    - L: Python scalar giving the output of the graph
    - grads: A tuple (grad_x1, grad_w1, grad_x2, grad_w2, grad_b, grad_y)
    giving the derivative of the output L with respect to each input.
    """
    # Forward pass: compute loss
    L = None
    a1 = x1*w1
    a2 = x2*w2
    y_hat = a1+a2+b
    d = y_hat - y
    L = d**2
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    # Backward pass: compute gradients
    grad_x1, grad_w1, grad_x2, grad_w2 = None, None, None, None
    grad_b, grad_y = None, None
    ###########################################################################
    # TODO: Implement the backward pass for the computational graph f1 shown  #
    # in the homework description. Store the gradients for each input         #
    # variable in the corresponding grad variables defined above.             #
    ###########################################################################
    grad_L = 1
    grad_d = (2*d)*grad_L
    grad_yHat = grad_d
    grad_y = -1*grad_d
    grad_a1 = grad_yHat
    grad_a2 = grad_yHat
    grad_b = grad_yHat
    grad_x1 = grad_a1*w1
    grad_w1 = grad_a1*x1
    grad_x2 = grad_a2*w2
    grad_w2 = grad_a2*x2
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_x1, grad_w1, grad_x2, grad_w2, grad_b, grad_y)
    return L, grads


def f2(x):
    """
    Computes the forward and backward pass through the computational graph f2
    from the homework PDF.

    A few clarifications about this graph:
    - The "x2" node multiplies its input by the constant 2
    - The "+1" and "-1" nodes add or subtract the constant 1
    - The division node computes y = t / b

    Inputs:
    - x: Python float

    Returns a tuple of:
    - y: Python float
    - grads: A tuple (grad_x,) giving the derivative of the output y with
      respect to the input x
    """
    # Forward pass: Compute output
    y = None
    ###########################################################################
    # TODO: Implement the forward pass for the computational graph f2 shown   #
    # in the homework description. Store the output in the variable y.        #
    ###########################################################################
    d = 2*x
    e = math.exp(d)
    t = e-1
    b = e+1
    y = t/b
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    # Backward pass: Compute gradients
    grad_x = None
    ###########################################################################
    # TODO: Implement the backward pass for the computational graph f2 shown  #
    # in the homework description. Store the gradients for each input         #
    # variable in the corresponding grad variables defined above.             #
    ###########################################################################
    grad_y = 1
    grad_t = grad_y*(1/b)
    grad_b = grad_y*((-1*t)/b**2)
    grad_e = grad_t+grad_b
    grad_d = grad_e*math.exp(d)
    grad_x = 2*grad_d
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y, (grad_x,)


def f3(s1, s2, y):
    """
    Computes the forward and backward pass through the computational graph f3
    from the homework PDF.

    A few clarifications about the graph:
    - The input y is an integer with y == 1 or y == 2; you do not need to
      compute a gradient for this input.
    - The division nodes compute p1 = e1 / d and p2 = e2 / d
    - The choose(p1, p2, y) node returns p1 if y is 1, or p2 if y is 2.

    Inputs:
    - s1, s2: Python floats
    - y: Python integer, either equal to 1 or 2

    Returns a tuple of:
    - L: Python scalar giving the output of the graph
    - grads: A tuple (grad_s1, grad_s2) giving the derivative of the output L
    with respect to the inputs s1 and s2.
    """
    assert y == 1 or y == 2
    # Forward pass: Compute loss
    L = None
    ###########################################################################
    # TODO: Implement the forward pass for the computational graph f3 shown   #
    # in the homework description. Store the loss in the variable L.          #
    ###########################################################################
    e1 = math.exp(s1)
    e2 = math.exp(s2)
    d = e1+e2
    p1 = e1/d
    p2 = e2/d
    if y == 1:
        pplus = p1
    else:
        pplus = p2
    L = -1*math.log(pplus)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    # Backward pass: Compute gradients
    grad_s1, grad_s2 = None, None

    grad_L = 1
    grad_pp = grad_L*-1*(1/pplus)
    grad_d = None
    if y == 1:
        grad_d = grad_pp*-1*(e1/d**2)
    else:
        grad_d = grad_pp * -1 * (e2 / d ** 2)
    grad_e1 = grad_d
    grad_e2 = grad_d
    grad_s1 = math.exp(s1)*grad_d
    grad_s2 = math.exp(s2)*grad_d
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_s1, grad_s2)
    return L, grads


def f3_y1(s1, s2):
    """
    Helper function to compute f3 in the case where y = 1

    Inputs:
    - s1, s2: Same as f3

    Outputs: Same as f3
    """
    return f3(s1, s2, y=1)


def f3_y2(s1, s2):
    """
    Helper function to compute f3 in the case where y = 2

    Inputs:
    - s1, s2: Same as f3

    Outputs: Same as f3
    """
    return f3(s1, s2, y=2)


def f4(x, y):
    loss, grads = None, None
    ###########################################################################
    # TODO: Implement a forward and backward pass through a computational     #
    # graph of your own construction. It should have at least five operators. #
    # Include a drawing of your computational graph in your report.           #
    # You can modify this function to take any number of arguments.           #
    ###########################################################################
    e1 = x**3
    p1 = e1+y
    e2 = p1**3
    p2 = e2+1
    loss = math.sqrt(p2)

    grad_L = 1
    grad_p2 = grad_L*(1/(2*math.sqrt(p2)))
    grad_e2 = grad_p2
    grad_p1 = (3*p1**2)*grad_e2
    grad_e1 = grad_p1
    grad_y = grad_p1
    grad_x = (3*x**2)*grad_e1
    grads = (grad_x, grad_y)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    return loss, grads
