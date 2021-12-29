import numpy as np
from openopt import NLP
from matplotlib.pyplot import figure, show


def meshtruss(p1: tuple, p2: tuple, nx: int, ny: int):
    """Creates the nodes and vertices (in a very stupid and inefficient way.)
    :param p1: tuple containing start coordinates, e.g. the graph origin
    :param p2: tuple containing end coordinates, e.g. graph edge
    :param nx: integer denoting amount of nodes in 1 row
    :param ny: integer denoting amount of rows with nodes
    """
    # Create x and y array coord-basis:
    xx = np.linspace(p1[0], p2[0], nx + 1)
    yy = np.linspace(p1[1], p2[1], ny + 1)
    # Actually create the Node coords:
    nodes = []
    for y in yy:
        for x in xx:
            nodes.append([x, y])

    # Create vertices between nodes:
    bars = []
    for j in range(ny):  # this is quite stupid, we are looping over the rows & nodes, we could do this much better.
        for i in range(nx):  # will need to edit something in here to stop vertical & horizontal vertices
            n1 = i + j * (nx + 1)  # n1 = y_index + x_index * (x_max +1); n1=5+3*(3+1)=17
            n2 = n1 + 1  # n2 = n1+1; e.g. 17+1=18 #what does this do???
            n3 = n1 + nx + 1  # n3 = n1 +x_max+1; e.g. 17+3+1 = 21
            n4 = n3 + 1  # n4 = n3+1; e.g. 21+1 = 22 #What does this do???
            bars.extend([[n1, n2], [n1, n3], [n1, n4], [n2, n3]])  # for each node, we extend bars with these n thingies
        bars.append([n2, n4]) # after each row, we append another [n2,n4] array. Still don't know what this is.
    index = ny * (nx + 1) + 1 # index = amount of row * (nodes per row +1) +1; e.g. 5 rows, 3 nodes --> 5*(3+1)+1 = 21
    for j in range(nx):  # WTF is this shit you can't make this up WHY ARE YOU USING J WHILE IT WAS I EARLIER!?
        bars.append([index + j - 1, index + j])  # after all this appending, we also add [index+x_index - 1, index+x_index]
        # e.g. [21+3-1, 21+3] = [23, 24]

    # might be much better to rewrite this entire function into a logical way to create bars.
    # I do not understand wtf they are doing with the n thingies.
    return np.array(nodes), np.array(bars)


def create_nodes_vertices(y_max, x_max, x_steps, y_steps, displacement):
    """Tom: new function to create nodes and vertices. creates nodes and vertices between them
    :param y_max: rows of stones in wall
    :param x_max: stones (nodes) per row
    :param x_steps: x distance between two nodes
    :param y_steps: y distance between two rows
    :param displacement: alternating displacement per row"""
    pass  # lets not implement this before understanding the other code


def opttruss(coord, connec, E, F, freenode, V0, plotdisp=False):
    n = connec.shape[0]  # num of vertices
    m = coord.shape[0]  # num of nodes ( = rows+1 * nodes/row+1; e.g. (6,4) -> 7*5 = 35
    vectors = coord[connec[:, 1], :] - coord[connec[:, 0], :]  # wut?
    l = np.sqrt((vectors ** 2).sum(axis=1))  # l: length?
    e = vectors.T / l # T = transposed array / l
    B = (e[np.newaxis] * e[:, np.newaxis]).T # not sure what this is?

    def fobj(x):
        D = E * x / l
        kx = e * D
        K = np.zeros((2 * m, 2 * m))
        for i in range(n):
            aux = 2 * connec[i, :]
            index = np.r_[aux[0]:aux[0] + 2, aux[1]:aux[1] + 2]
            k0 = np.concatenate((np.concatenate((B[i], -B[i]), axis=1),
                                 np.concatenate((-B[i], B[i]), axis=1)), axis=0)
            K[np.ix_(index, index)] = K[np.ix_(index, index)] + D[i] * k0

        block = freenode.flatten().nonzero()[0]
        matrix = K[np.ix_(block, block)]
        rhs = F.flatten()[block]
        solution = np.linalg.solve(matrix, rhs)
        u = freenode.astype('float').flatten()
        u[block] = solution
        U = u.reshape(m, 2)
        axial = ((U[connec[:, 1], :] - U[connec[:, 0], :]) * kx.T).sum(axis=1)
        stress = axial / x
        cost = (U * F).sum()
        dcost = -stress ** 2 / E * l
        return cost, dcost, U, stress

    def volume(x):
        return (x * l).sum(), l

    def drawtruss(x, factor=3, wdt=5e2):
        U, stress = fobj(x)[2:]
        newcoor = coord + factor * U
        if plotdisp:
            fig = figure(figsize=(12, 6))
            ax = fig.add_subplot(121)
            bx = fig.add_subplot(122)
        else:
            fig = figure()
            ax = fig.add_subplot(111)
        for i in range(n):
            bar1 = np.concatenate((coord[connec[i, 0], :][np.newaxis],
                                   coord[connec[i, 1], :][np.newaxis]), axis=0)
            bar2 = np.concatenate((newcoor[connec[i, 0], :][np.newaxis],
                                   newcoor[connec[i, 1], :][np.newaxis]), axis=0)
            if stress[i] > 0:
                clr = 'r'
            else:
                clr = 'b'
            ax.plot(bar1[:, 0], bar1[:, 1], color=clr, linewidth=wdt * x[i])
            ax.axis('equal')
            ax.set_title('Stress')
            if plotdisp:
                bx.plot(bar1[:, 0], bar1[:, 1], 'r:')
                bx.plot(bar2[:, 0], bar2[:, 1], color='k', linewidth=wdt * x[i])
                bx.axis('equal')
                bx.set_title('Displacement')
        show()

    xmin = 1e-6 * np.ones(n)
    xmax = 1e-2 * np.ones(n)
    f = lambda x: fobj(x)[0]  # NO NO NO NO NO NO NO WHYYYYYYYYYYYYYYYYYYYYYYY
    derf = lambda x: fobj(x)[1]  # This is not how you should use lambda functions
    totalvolume = volume(xmax)[0]
    constr = lambda x: 1. / totalvolume * volume(x)[0] - V0
    dconstr = lambda x: 1. / totalvolume * volume(x)[1]
    x0 = 1e-4 * np.ones(n)
    problem = NLP(f, x0, df=derf, c=constr, dc=dconstr, lb=xmin, ub=xmax, name='Truss', iprint=100)
    result = problem.solve("mma")

    drawtruss(result.xf)


def remove_bar(connec, n1, n2):
    # if we remove a connection we should also remove a node?
    bars = connec.tolist()
    for bar in bars[:]:
        if (bar[0]==n1 and bar[1] == n2) or (bar[0] == n2 and bar[1] == n1):
            bars.remove(bar)
            return np.array(bars)
    else:
        print("Bar does not exist")
        return connec


def remove_node(connec, n1):
    # if we remove a node we should also remove its connections?
    bars = connec.tolist()
    for bar in bars[:]:
        if bar[0] == n1 or bar[1] == n1:
            bars.remove(bar)
    return np.array(bars)