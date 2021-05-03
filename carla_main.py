import qr_trainer

import os


Agent1 = 1
logging_dir = "carla01"

training = False
parallel_training = False
ttc_list = [0.5, 0.75, 1]
test_ttc_list = [0.4]  # [0,0.5,1,1.5,2,2.5,3,3.5]
jobs = []
if not os.path.exists('logs/' + logging_dir + '/network/save'):
    os.makedirs('logs/' + logging_dir + '/network/save')
    os.makedirs('logs/' + logging_dir + '/network/load')
    os.makedirs('logs/' + logging_dir + '/figures')
    os.makedirs('logs/' + logging_dir + '/speeds')

n_actions = 5
n_stacked_frames = 4
n_frame_dropout = 5
uncertainty_enabled = True

if uncertainty_enabled:
    input_dims = [n_stacked_frames + 1, 84, 84]
else:
    input_dims = [n_stacked_frames, 84, 84]

gamma = 0.99
lr = 0.00001

batch_size = 32
replace = 1000
mem_size = 20_000

epsilon = 0.5
eps_min = 0.01
eps_dec = 1e-5

control_dropout_enabled = True
train_network = True
load_network = False

n_actions = 3
n_stacked_frames = 4
n_frame_dropout = 4
if uncertainty_enabled:
    input_dims = [n_stacked_frames + 1, 84, 84]
else:
    input_dims = [n_stacked_frames, 84, 84]

gamma = 0.99
lr = 0.0001

batch_size = 64
replace = 5_000
mem_size = 50_000
eps_dec = 1e-5
n_quants = 32

n_episodes = 1000

uncer_str = 'risk_aware'
algo_name = 'qrdqn'

control_dropout_enabled = False
def main():
    #dq = dqn.trainer()
    #dq.train()
    dq =qr_trainer.trainer(gamma=gamma,
                                      epsilon=epsilon,
                                      lr=lr,
                                      n_actions=n_actions,
                                      uncertainty_enabled=uncertainty_enabled,
                                      mem_size=mem_size,
                                      batch_size=batch_size,
                                      input_dimention=input_dims,
                                      n_stacked_frames=n_stacked_frames,
                                      n_frame_dropout=n_frame_dropout,
                                      eps_min=eps_min,
                                      eps_dec=eps_dec,
                                      replace=replace,
                                      control_dropout_enabled=control_dropout_enabled,
                                      load_network=load_network,
                                      train_network=train_network,
                                      n_episodes=n_episodes,
                                      remove_side=False,
                                      chkpt_name=f"{algo_name}_{uncer_str}",
                                      ttc=0.5,
                                      logging_dir=logging_dir,
                                      n_quants=10
                                      )
    dq.train()


if __name__ == '__main__':

    #cProfile.run('main()')
    main()
