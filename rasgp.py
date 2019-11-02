import numpy.matlib
import os
from functions_v3 import *
import multiprocessing as mp
import sys


def job(params):
    np.random.seed()

    # Create the network
    n = params['n']
    x0 = params['x0']
    d = params['d']
    g = params['g']
    nodes = []
    for _ in range(n):
        nodes.append(node_class(x0, d))

    for i in range(n):
        for j in g.neighbors(i):
            nodes[i].add_out_nei(nodes[j])

    output = rasgp(nodes, params)
    return output


def parallel(n_task, n_proc, params):
    pool = mp.Pool(processes=n_proc)
    res_pool = [pool.apply_async(job, (params,)) for _ in range(n_task)]

    ite_max = params['ite_max']
    results = []
    E_opt_sum = np.zeros(ite_max)
    E_con_sum = np.zeros(ite_max)
    E2_opt_sum = np.zeros(ite_max)
    suc_sim = 0
    for res in res_pool:
        output = res.get()
        if not output['flag']:
            suc_sim += 1
            results.append(output)
            E_opt_sum += output['E_opt']
            E2_opt_sum += output['E_opt']**2
            E_con_sum += output['E_con']
        else:
            print 'one simulation failed and discarded.'

    pool.close()
    pool.join()
    return {'suc_sim': suc_sim, 'E_opt_sum': E_opt_sum, 'E2_opt_sum': E2_opt_sum,
            'E_con_sum': E_con_sum, 'results': results}


if __name__ == '__main__':
    # Main Settings
    setting = 1  # int(sys.argv[2]) # <----
    if setting == 1:
        pw = 1.0
        pf = 0.0
        lf = 0
        ld = 0
        lu = 1
        g_type = 'c'

    elif setting == 2:
        pw = 1.0
        pf = 0.3
        lf = 3
        ld = 3
        lu = 1
        g_type = 'c'

    elif setting == 3:
        pw = 0.5
        pf = 0.3
        lf = 3
        ld = 3
        lu = 3
        g_type = 'c'

    elif setting == 4:
        pw = 0.5
        pf = 0.3
        lf = 3
        ld = 3
        lu = 3
        g_type = 'r'

    else:
        raise Exception('Setting not detected.')

    # Parallel Settings
    n_proc = 7  # int(os.getenv('NSLOTS'))  # <----
    n_task = n_proc * 1
    load_sum = False  # <----

    ite_max = 20000
    k0 = 100
    step = 100
    reps = 300 / n_task  # <----

    n = 10  # int(sys.argv[1])  # <----
    m = 50  # do not change

    # Function Settings (do not change)
    d = 3
    mu = 1.0
    ns = 4.0
    c = 5.0 / n

    # h = lu * d * n * (ns ** 2) / (12 * (mu ** 2))
    # Load Data
    Abs, Ab = load_data('Data/Data_n100x%d.pkl' % m)
    Ab = Ab[:n * m]

    x0 = np.matlib.ones((1, d))
    xo = get_optimal(Ab, c)  # <----

    if g_type == 'c':
        g_name = '/cyc_'  # <----
        g, _ = gen_cycle(n, bidi=True)  # <----
    elif g_type == 'r':
        g_name = '/ran_'
        g, _ = gen_digraph(n, 0.5)
    else:
        raise Exception('Graph type not detected!')

    params = {'n': n, 'd': d, 'mu': mu, 'ns': ns, 'c': c, 'step': step,
              'pw': pw, 'pf': pf, 'lf': lf, 'ld': ld, 'lu': lu, 'Abs': Abs,
              'xo': xo, 'x0': x0, 'k0': k0, 'ite_max': ite_max, 'g': g}

    # Save settings
    save_path = "rasgp"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # load summary file
    # E_opts = []
    # E2_opts = []
    # E_cons = []
    E_opt_sum = np.zeros(ite_max)
    E_con_sum = np.zeros(ite_max)
    E2_opt_sum = np.zeros(ite_max)
    rep_all = 0.0

    file_name = g_name + 'n%d_ns%d_lu%d_ld%d_lf%d_%dk.pkl' % (n, ns, lu, ld, lf, ite_max / 1000)
    if os.path.isfile(save_path + file_name) and load_sum:
        with open(save_path + file_name, 'rb') as f:
            data = pickle.load(f)
        f.close()

        for key in data.keys():
            vars()[key] = data[key]
        del data
        print 'Previous data loaded successfully.'

    exe_avg = 0.0
    for rep in range(reps):
        print '----'
        print 'Starting rep = %d / %d' % (rep, reps)
        t_s = time.time()  # start time

        pool_output = parallel(n_task, n_proc, params)
        # {'suc_sim', 'E_opt_sum', 'E2_opt_sum', 'E_con_sum', 'results'}

        suc_sim = pool_output['suc_sim']
        # E_opts.append(compress(pool_output['E_opt_sum']/suc_sim,step))
        # E2_opts.append(compress(pool_output['E2_opt_sum']/suc_sim,step))
        # E_cons.append(compress(pool_output['E_con_sum']/suc_sim,step))
        E_opt_sum += pool_output['E_opt_sum']
        E2_opt_sum += pool_output['E2_opt_sum']
        E_con_sum += pool_output['E_con_sum']
        rep_all += suc_sim

        exe_p = time.time() - t_s  # execution time of one parallel call
        print 'execution of {} tasks: {:.2f} (sec).'.format(n_task, exe_p)

        exe_avg = (exe_avg * rep + exe_p) / (rep + 1)
        print 'average execution time ~ {:.2f} (sec)'.format(exe_avg)

        print 'remaining time ~ {:.1f} minute(s)'.format((reps - rep - 1) * exe_avg / 60)

        # save!
        with open(save_path + file_name, 'wb') as f:
            pickle.dump({'E_opt_sum': E_opt_sum, 'E2_opt_sum': E2_opt_sum, 'E_con_sum': E_con_sum,
                         'rep_all': rep_all, 'params': params, 'g': g}, f)

        print 'data saved! total # of simulations: %d' % rep_all

        E_opt_avg = E_opt_sum/rep_all
        E2_opt_avg = E2_opt_sum/rep_all
        draw_fig(E_opt_avg, E2_opt_avg, rep_all)

    print('Simulation Finished!')


