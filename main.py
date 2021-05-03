
import test_script
import os
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import sys
import glob
try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
    print('didnt read here')


def main():
    logging_dir = "train_01_carla"

    training = False
    parallel_training = False
    ttc_list = [0,0.25,0.5]
    test_ttc_list = [0,0.5,0.75,1]#[0,0.5,1,1.5,2,2.5,3,3.5]
    jobs = []
    if not os.path.exists('logs/' + logging_dir+'/network/save'):
        os.makedirs('logs/' + logging_dir+'/network/save')
        os.makedirs('logs/' + logging_dir + '/network/load')
        os.makedirs('logs/' + logging_dir + '/figures')
        os.makedirs('logs/' + logging_dir + '/speeds')

    if not training:

        algos = [2]
        n_test_episodes = 1000
        uncertainty_list = [True]
        for algo_n in algos:

            for uncertainty in uncertainty_list:
                tester = test_script.Tester(algo_n,
                                            n_test_episodes,
                                            uncertainty,
                                            logging_dir=logging_dir,
                                            train_network=training)
                tester.ttc_loop(test_ttc_list)
                tester.plot()


    else:
        if parallel_training:

            algos = [0]

            n_test_episodes = 1000
            uncertainty_list = [True]
            for algo_n in algos:

                for uncertainty in uncertainty_list:

                    p = mp.Process(target=train_process,args=(algo_n,uncertainty))
                    jobs.append(p)
                    p.start()

        else:
            algos = [3]

            n_test_episodes = 5_000
            uncertainty_list = [True]
            for algo_n in algos:

                for uncertainty in uncertainty_list:
                    tester = test_script.Tester(algo_n, n_test_episodes, uncertainty,
                                                logging_dir=logging_dir,
                                                train_network=training)
                    tester.train_all(ttc_list)


def train_process(algo,uncertainty):
    n_test_episodes = 1500
    logging_dir = "train05"
    ttc_list = [1, 1.5, 2, 2.5, 3, 3.5]
    tester = test_script.Tester(algo, n_test_episodes, uncertainty, logging_dir=logging_dir)
    tester.train_all(ttc_list)




if __name__ == '__main__':

    main()
