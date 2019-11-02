import numpy as np
import random
import networkx as nx
from sklearn.svm import LinearSVC
import pickle
import time
import matplotlib.pyplot as plt

def load_data(D_path):
    with open(D_path, 'r') as f:
        Data = pickle.load(f)
    f.close()
    Ab = Data['Ab']
    Abs = Data['Abs']
    del Data  # to free up some space!
    return Abs, Ab


def get_edges(A, n):
    edges = []
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                edges.append((i,j))
    return edges


def gen_digraph(n, thresh):
    while True:
        A = np.random.rand(n,n)
        A -= np.eye(n)
        A[A < thresh] = 0
        A[A >= thresh] = 1
        A = A.astype(int, copy=False)
        g = nx.DiGraph()
        g.add_edges_from(get_edges(A, n))
        if g.number_of_nodes() == n:
            if nx.is_strongly_connected(g):
                break
    Ag = nx.adjacency_matrix(g)
    return g, Ag


def circle(n, bidi = False):
    E = []
    for i in range(n-1):
        E.append((i,i+1))
        if bidi:
            E.append((i+1,i))
    E.append((n-1,0))
    if bidi:
        E.append((0,n-1))
    return E


def gen_cycle(n, bidi=False):
    g = nx.DiGraph()
    g.add_edges_from(circle(n, bidi))
    Ag = nx.adjacency_matrix(g)
    return g, Ag


def gen_grid(m):
    g = nx.grid_graph([m,m])
    Ag = nx.adjacency_matrix(g)
    nx.draw(g, with_labels=True)
    plt.show()
    print(nx.info(g))
    return g, Ag


def hinge_der(x):
    if x<=0:
        return -1.0
    if x<=1 and x>0:
        return x-1.0
    else:
        return 0.0


hd = np.vectorize(hinge_der, otypes=[np.float])


def get_grad(Ab, z):
    b = np.matrix(Ab[:,-1])
    Ae = Ab.copy()
    Ae[:,-1] = 1.0
    ksi = np.multiply(b.T,np.dot(Ae,z.T))
    return np.dot(Ae.T,np.multiply(b.T,hd(ksi))).T


def get_optimal(Ab, c):
    thresh = 1e-8

    clf = LinearSVC(loss='hinge', C=c)
    clf.fit(Ab[:,:2], Ab[:,2])
    xo = np.append(clf.coef_, clf.intercept_)
    xo = np.matrix(xo)

    k = 1
    kmax_opt = 50000
    while k < kmax_opt:
        k += 1
        go = xo + c * get_grad(Ab, xo)
        alpha = 1.0 /k

        xo -= alpha * go
        if np.linalg.norm(go) < thresh:
            print 'Optimal Reached!'
            break

    if k == kmax_opt:
        print 'Max iter. reached!'
    return xo


class node_class(object):
    def __init__(self, x, d):
        self.d = d
        self.x = x + 0
        self.y = 1.0
        self.z = self.x/self.y

        self.sigma_x = np.matlib.zeros((1, self.d))
        self.sigma_y = 0.0

        self.rhos_x = {}
        self.rhos_y = {}
        self.kappa = {}

        self.inbox = []
        self.fails = {}
        self.d_out = 0.0

        self.sleep = 0

    def add_out_nei(self, nei):
        if nei in self.fails.keys():
            print 'Neighbor already exists'
        else:
            nei.rhos_x[self] = np.matlib.zeros((1, self.d))
            nei.rhos_y[self] = 0.0
            nei.kappa[self] = -1

            self.fails[nei] = 0
            self.d_out += 1

    def broadcast(self, pf, lf, ld, k):
        self.x = self.x/(self.d_out + 1.0)
        self.sigma_x += self.x

        self.y = self.y/(self.d_out + 1.0)
        self.sigma_y += self.y

        for j in self.fails.keys():
            rf = random.random()
            if (rf <= pf) and (self.fails[j] < lf):
                self.fails[j] += 1
            else:
                self.fails[j] = 0
                delay = random.randint(1, ld + 1)
                msg = (self, k, k + delay, self.sigma_x + 0.0, self.sigma_y + 0.0)
                j.inbox.append(msg)

    def update(self, step):
        self.x = self.x + step

    def process(self, k):
        m = len(self.inbox)
        i = 0
        while i < m:
            msg = self.inbox[i]

            j = msg[0]
            if msg[1] < self.kappa[j]:
                self.inbox.remove(msg)
                m -= 1
            elif msg[2] <= k:
                self.kappa[j] = msg[1]
                self.x += msg[3] - self.rhos_x[j]
                self.rhos_x[j] = msg[3] + 0.0

                self.y += msg[4] - self.rhos_y[j]
                self.rhos_y[j] = msg[4] + 0.0

                self.inbox.remove(msg)
                m -= 1
            else:
                i += 1
        self.z = self.x/self.y


def rasgp(nodes, params):
    start_time = time.time()
    n = params['n']
    d = params['d']
    c = params['c']
    mu = params['mu']
    ns = params['ns']
    Abs = params['Abs']
    xo = params['xo']
    lu = params['lu']
    lf = params['lf']
    ld = params['ld']
    pw = params['pw']
    pf = params['pf']
    k0 = params['k0']
    ite_max = params['ite_max']
    thresh_y = 5e-8

    Zs = np.matlib.zeros((n,d))
    for i in range(n):
        Zs[i] = nodes[i].z
    E_opt = []
    E_con = []

    for k in range(k0, k0 + ite_max):
        if k > 0:
            alpha = (n / (mu * k))
        else:
            alpha = 0.0

        for i in range(n):
            node = nodes[i]
            g = node.z / n + c * get_grad(Abs[i], node.z) + ns * (np.random.random(d) - 0.5)
            node.update(-alpha * g)

            rw = random.random()  # wake up r.v.
            if (rw > pw) and (node.sleep < lu - 1):
                node.sleep += 1
            else:
                node.sleep = 0
                node.broadcast(pf, lf, ld, k)
                node.process(k)
                if node.y < thresh_y:
                    return {'flag': True}
                Zs[i] = node.z


        z_bar = np.mean(Zs, axis=0)
        E_con.append((np.linalg.norm(Zs - z_bar) ** 2) / n)
        E_opt.append(np.linalg.norm(z_bar - xo) ** 2)

    exe = time.time() - start_time
    return {'E_opt': np.array(E_opt), 'E_con': np.array(E_con), 'exe': exe, 'z_bar': z_bar, 'flag': False}


def main_cen(params):
    start_time = time.time()

    SU = params['SU']
    xo = params['xo']
    x0 = params['x0']
    k0 = params['k0']
    ite_max = params['ite_max']
    c = params['c']
    d = params['d']
    Ab = params['Ab']
    n = params['n']
    mu = params['mu']
    ns = params['ns']

    xc = x0.copy()
    E_opt_c = []  # [np.linalg.norm(xc - xo)**2]

    k = k0

    while k < ite_max + k0:
        k += 1

        gc = xc + c * get_grad(Ab, xc)
        if SU:
            gc += ns * (np.random.random(d) - 0.5) * n
        else:
            gc += ns * (np.random.random(d) - 0.5) * np.sqrt(n)

        alpha = 1.0 / (mu * k)
        xc -= alpha * gc

        E_opt_c.append(np.linalg.norm(xc - xo) ** 2)

    end_time = time.time()
    exe = end_time - start_time

    return {'E_opt': np.array(E_opt_c), 'xc': xc, 'exe': exe}


def max_future(errors):
    k = len(errors)
    output = np.zeros(k)
    max_error = 0.0
    for k in range(-1,-k-1,-1):
        if errors[k] > max_error:
            max_error = errors[k]
        output[k] = max_error
    return output


def compress(arr, step):
    arr_comp = []
    l_arr = len(arr)
    i = 1
    while i*step <= l_arr:
        arr_comp.append(np.mean(arr[(i-1)*step:i*step]))
        i += 1
    return arr_comp


def draw_fig(E_opt_avg, E2_opt_avg, rep_all):
    E_opt_avg = np.array(E_opt_avg)
    E2_opt_avg = np.array(E2_opt_avg)
    E_opt_var = E2_opt_avg - E_opt_avg**2
    E_opt_std = np.sqrt(E_opt_var) / np.sqrt(rep_all)
    # E_opt_avg_maxed = max_future(E_opt_avg)
    plt.plot(range(len(E_opt_avg)), E_opt_avg, linewidth=1, label='Error')
    plt.fill_between(range(len(E_opt_avg)), E_opt_avg - E_opt_std, E_opt_avg + E_opt_std, alpha=0.4)
    # plt.plot(range(len(E_opt_avg_maxed)), E_opt_avg_maxed, linewidth=1, label='Error Maxed')
    plt.title('%d Simulations' % rep_all)
    plt.xlabel('iteration (k)')
    plt.ylabel('Optimization Error')
    plt.legend(loc='upper right')
    plt.grid()
    plt.axis([0, len(E_opt_avg), 0, E_opt_avg[-1]*4])
    plt.show()
