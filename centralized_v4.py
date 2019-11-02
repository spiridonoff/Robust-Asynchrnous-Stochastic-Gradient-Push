import numpy.matlib
import os
from functions_v3 import *
import multiprocessing as mp
import sys


# Parallel:
def job(params):  # Job of each processor
    np.random.seed()
    output = main_cen(params)
    return output


def parallel(n_task, n_proc, params): # creates the pool of processors and collects their outputs
    pool = mp.Pool(processes=n_proc)
    res_pool = [pool.apply_async(job, (params,)) for _ in range(n_task)]

    results = []
    ite_max = params['ite_max']
    E_opt_sum = np.zeros(ite_max)
    E2_opt_sum = np.zeros(ite_max)

    for res in res_pool:
        output = res.get()
        results.append(output)
        E_opt_sum += output['E_opt']
        E2_opt_sum += output['E_opt']**2
    pool.close()
    pool.join()
    return {'E_opt_sum': E_opt_sum, 'E2_opt_sum': E2_opt_sum, 'results': results}


if __name__ == '__main__':
    n = 50  # int(sys.argv[1]) # Network Size
    m = 50  # Data size of each agent

    # Parallel Settings
    n_proc = 7    # int(os.getenv('NSLOTS'))  # <----
    n_task = n_proc * 1
    load_sum = False  # <----

    # Main settings
    ite_max = 20000  # number of iterations
    k0 = 100  # Initial k
    reps = 300 / n_task  # <----  # number of simulations

    # Objective Function Setting (DO NOT CHANGE) ####
    d = 3  # dimension of the data points
    c = 5.0 / n  # penalty term of SVM
    mu = 1.0  # strong convexity of the sum of local functions
    ns = 4.0  # Noise Support
    SU = True  # Speed-up? Affectes the variance of the noise added

    # Load Data
    _, Ab = load_data('Data/Data_n100x%d.pkl' % m)
    Ab = Ab[:n * m]

    x0 = np.matlib.ones((1, d))
    xo = get_optimal(Ab, c)
    params = {'n': n, 'd': d, 'mu': mu, 'ns': ns, 'c': c, 'Ab': Ab, 'SU': SU,
              'xo': xo, 'x0': x0, 'k0': k0, 'ite_max': ite_max, 'step': step}

    # Save settings
    save_path = "res_cen"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    E_opt_sum = np.zeros(ite_max)
    E2_opt_sum = np.zeros(ite_max)
    rep_all = 0.0

    file_name = '/cen_n%d_ns%d_%dk' % (n, ns, ite_max / 1000)
    if SU:
        file_name += '_su.pkl'
    else:
        file_name += '.pkl'

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
        print 'Starting rep = %d / %d' %(rep,reps)
        t_s = time.time()  # start time

        pool_output = parallel(n_task, n_proc, params)

        E_opt_sum += pool_output['E_opt_sum']
        E2_opt_sum += pool_output['E2_opt_sum']
        rep_all += n_task

        exe_p = time.time() - t_s  # execution time of one parallel call
        print 'execution of {} tasks: {:.2f} (sec).'.format(n_task, exe_p)

        exe_avg = (exe_avg * rep + exe_p) / (rep+1)
        print 'average execution time ~ {:.2f}'.format(exe_avg)

        print 'remaining time ~ {:.1f} minute(s)'.format((reps - rep - 1) * exe_avg / 60)

        # save!
        with open(save_path + file_name, 'wb') as f:
            pickle.dump({'E_opt_sum': E_opt_sum, 'E2_opt_sum': E2_opt_sum,
                         'rep_all': rep_all, 'params': params}, f)

        print 'data saved! total # of simulations: %d' % rep_all

        E_opt_avg = E_opt_sum / rep_all
        E2_opt_avg = E2_opt_sum / rep_all
        draw_fig(E_opt_avg, E2_opt_avg, rep_all)
        draw_fig(compress(E_opt_avg,step), compress(E2_opt_avg,step), rep_all)
    print('Simulation Finished!')



