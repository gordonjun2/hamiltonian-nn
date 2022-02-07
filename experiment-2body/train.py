# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP, MLP_sgp4
from hnn import HNN
from data import get_dataset, get_sgp4_orbit, coords2state_sgp4
from utils import L2_loss, to_pickle, from_pickle, get_model_parm_nums

from tensorboardX import SummaryWriter       # importing tensorboard
import random
import math

# for integrating a vector field parameterized by a NN or HNN
def model_update(t, state, model, device):
    
    state = state.reshape(-1,7)

    deriv = np.zeros_like(state)
    np_x = state[:,1:] # drop mass
    np_x = np_x.T.flatten()[None, :]
    x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).to(device)
    dx_hat = model.time_derivative(x)
    deriv[:,1:] = dx_hat.detach().data.cpu().numpy().reshape(6,2).T
    return deriv.reshape(-1)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('--input_dim', default=2*4, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')   # original default is 200
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')          # original default is 1e-3
    parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
    parser.add_argument('--input_noise', default=0.0, type=int, help='std of noise added to inputs')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=1, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='2body', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn (solenoidal or conservative)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--exp_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu_enable', dest='gpu_enable', action='store_true', help='include if gpu is to be used')
    parser.add_argument('--gpu_select', default=0, type=int, help='select which gpu to use')
    parser.add_argument('--satellite_problem', dest='satellite_problem', action='store_true', help='set scenario to be Satellite Problem instead of Two-Body Problem as demonstrated in the paper')
    parser.add_argument('--data_percentage_usage', default=1, type=float, help='percentage of data to use (eg. 1 means all data)')
    parser.add_argument('--save_best_weights', dest='save_best_weights', action='store_true', help='to save weight if result is better than before')
    parser.add_argument('--epoch', default=1, type=int, help='epoch for satellite_problem')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args, save_dir, tb, label, test_split=0.2):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # initialise device
  device = torch.device('cuda:' + str(args.gpu_select) if args.gpu_enable else 'cpu')

  # parameters to test
  hidden_dim_list = [200, 300]
  learn_rate_list = [1e-4, 1e-3, 1e-2]
  #batch_size_list = [300]
  input_noise_list = [0, 0.05, 0.1]
  layers_list = [3, 4]
  
  # initialise infinity loss first
  best_test_loss = float('inf')
    
  trial_no = 0

  for hidden_dim in hidden_dim_list:
    for learn_rate in learn_rate_list:
      for input_noise in input_noise_list:
        for layers in layers_list:
        
          trial_no = trial_no + 1
   
          # init model and optimizer
          if args.verbose:
            print("Training baseline model:" if args.baseline else "Training HNN model:")
            print("hidden_dim: " + str(hidden_dim) + ", learn_rate: " + str(learn_rate) + ", input_noise: " + str(input_noise) + ", layers: " + str(layers))

          if args.satellite_problem:
            output_dim = 2*6 if args.baseline else 2
            if layers == 3:
                nn_model = MLP_sgp4(2*6, hidden_dim, output_dim, args.nonlinearity).to(device)
            else:
                nn_model = MLP(2*6, hidden_dim, output_dim, args.nonlinearity).to(device)
            model = HNN(2*6, differentiable_model=nn_model,
                    field_type=args.field_type, baseline=args.baseline, device = device)
          else:
            output_dim = 2*4 if args.baseline else 2
            nn_model = MLP(2*4, hidden_dim, output_dim, args.nonlinearity).to(device)
            model = HNN(2*4, differentiable_model=nn_model,
                    field_type=args.field_type, baseline=args.baseline, device = device)

          num_parm = get_model_parm_nums(model)

          print('model contains {} parameters'.format(num_parm))

          optim = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=0)

          # arrange data
          data = get_dataset(args.name, args.exp_dir, satellite_problem = args.satellite_problem, data_percentage_usage = args.data_percentage_usage, verbose=True)

          if args.satellite_problem:
            x = torch.tensor( data[0]['coords'], requires_grad=True, dtype=torch.float32).to(device)
            # Each line of 'x' is in the form (qx1, qx2, qy1, qy2, px1, px2, py1, py2) in original 2-body experiment and (qx1, qx2, qy1, qy2, qz1, qz2, px1, px2, py1, py2, pz1, pz2) in the satellite-problem experiment
            test_x = torch.tensor( data[0]['test_coords'], requires_grad=True, dtype=torch.float32).to(device)
            dxdt = torch.Tensor(data[0]['dcoords']).to(device)
            test_dxdt = torch.Tensor(data[0]['test_dcoords']).to(device)
            
            # a list of lengths of every trajectory
            lengths = data[1]['lengths']
            print('lengths: ', len(lengths))
            
          else:
            x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32).to(device)
            # Each line of 'x' is in the form (qx1, qx2, qy1, qy2, px1, px2, py1, py2) in original 2-body experiment and (qx1, qx2, qy1, qy2, qz1, qz2, px1, px2, py1, py2, pz1, pz2) in the satellite-problem experiment
            test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32).to(device)
            dxdt = torch.Tensor(data['dcoords']).to(device)
            test_dxdt = torch.Tensor(data['test_dcoords']).to(device)

          stats = {'train_loss': [], 'test_loss': []}
            
          if not args.satellite_problem:
              for step in range(args.total_steps+1):
                # train step

                # 'torch.randperm(x.shape[0])' randomizes array index (shuffling) and '[:args.batch_size]' slices the first 'batch_size' array index for training
                ixs = torch.randperm(x.shape[0])[:args.batch_size]
                dxdt_hat = model.time_derivative(x[ixs])
                dxdt_hat += input_noise * torch.randn(*x[ixs].shape).to(device) # add noise, maybe
    #             if args.verbose and step % args.print_every == 0:
    #                 print('\nExample Training Ground Truth: ', dxdt[ixs][0])
    #                 print('\nExample Training Prediction: ', dxdt_hat[0])
                print(dxdt_hat.shape)
                loss = L2_loss(dxdt[ixs], dxdt_hat)
                loss.backward()
                grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
                optim.step() ; optim.zero_grad()

                # run test data
                test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
                test_dxdt_hat = model.time_derivative(test_x[test_ixs])
                test_dxdt_hat += input_noise * torch.randn(*test_x[test_ixs].shape).to(device) # add noise, maybe
    #             if args.verbose and step % args.print_every == 0:
    #                 print('\nExample Testing Ground Truth: ', test_dxdt[test_ixs][0])
    #                 print('\nExample Testing Prediction: ', test_dxdt_hat[0])
                test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

                # logging
                stats['train_loss'].append(loss.item())
                stats['test_loss'].append(test_loss.item())

                tb.add_scalar('Train loss vs Steps', loss.item(), step)
                tb.add_scalar('Test loss vs Steps', test_loss.item(), step)

                if args.verbose and step % args.print_every == 0:
                  print("\nstep {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
                      .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))
                if args.save_best_weights and step % args.print_every == 0:
                    if test_loss.item() < best_test_loss:
                        path = '{}/{}-orbits-{}-step-{}.tar'.format(save_dir, args.name, label, step)
                        torch.save(model.state_dict(), path)
                        best_test_loss = test_loss.item()

              train_dxdt_hat = model.time_derivative(x)
              train_dist = (dxdt - train_dxdt_hat)**2
              test_dxdt_hat = model.time_derivative(test_x)
              test_dist = (test_dxdt - test_dxdt_hat)**2
              print('\nFinal train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}\n'
                .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
                        test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
            
          else:
              train_lengths_num = math.ceil(len(lengths) - len(lengths)*test_split)
              test_lengths_num = math.floor(len(lengths)*test_split)
              train_trajectory_start = test_lengths_num
                
              step = 0
            
              for epoch_no in range(args.epoch):
                for trajectory_no in range(train_lengths_num):
                    #trajectory_no = 0
                    # train step for position
                    
                    train_trajectory_end = train_trajectory_start + lengths[test_lengths_num + trajectory_no]
                    trajectory_states = x[train_trajectory_start:train_trajectory_end, :]
                    
                    print(train_trajectory_start)
                    print(train_trajectory_end)
                    
#                     t_points = 2
#                     t_span = [1, t_points]
#                     initial_coord = x[train_trajectory_start]
#                     state = coords2state_sgp4(initial_coord)
#                     update_fn = lambda t, y0: model_update(t, y0, model, device)
#                     hnn_orbit, settings = get_sgp4_orbit(state, t_points=t_points, t_span=t_span, update_fn=update_fn)
#                     hnn_orbit_pos = hnn_orbit[0].T[:, [1, 2, 3]]
                    
#                     next_true_pos = trajectory_states[:, [0, 2, 4]]
#                     print('Train Next True Pos: ', next_true_pos[1, :])
#                     next_predicted_pos = torch.tensor(hnn_orbit_pos).to(device)
#                     print('Train Next Predicted Pos: ', next_predicted_pos[1, :])
                    
                    
#                     if args.verbose and step % args.print_every == 0:
#                         print('\nExample Training Ground Truth: ', next_true_pos)
#                         print('\nExample Training Prediction: ', next_predicted_pos)
                  
#                     loss_pos = L2_loss(next_true_pos[1, :], next_predicted_pos[1, :])
#                     print('Train Loss Pos: ', loss_pos.item())
                
                    # train step for position derivatives
            
                    dxdt_hat = model.time_derivative(x[train_trajectory_start:train_trajectory_end, :])
                    dxdt_hat += input_noise * torch.randn(*x[train_trajectory_start:train_trajectory_end, :].shape).to(device) # add noise, maybe
                    #print(dxdt_hat)
                    
#                     if args.verbose and step % args.print_every == 0:
#                         print('\nExample Training Ground Truth: ', dxdt[ixs][0])
#                         print('\nExample Training Prediction: ', dxdt_hat[0])
        
                    loss_dxdt = L2_loss(dxdt[train_trajectory_start:train_trajectory_end, :], dxdt_hat)
                    #print('Train Loss dxdt: ', loss_dxdt.item())
        
                    #loss = loss_pos + loss_dxdt
                    #print('Train Loss: ', loss.item())
                    loss = loss_dxdt
                    loss.backward()
                    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
                    optim.step() ; optim.zero_grad()

                    # run test data
                    index = random.randint(0, test_lengths_num-1)
                    test_trajectory_start = sum(lengths[:index])
                    test_trajectory_end = test_trajectory_start + lengths[index]
                    
                    
                    # test step for position
                    
#                     trajectory_states = x[test_trajectory_start:test_trajectory_end, :]
                    
#                     t_points = test_trajectory_end - test_trajectory_start
#                     t_span = [1, t_points]
#                     initial_coord = x[test_trajectory_start]
#                     state = coords2state_sgp4(initial_coord)
#                     update_fn = lambda t, y0: model_update(t, y0, model, device)
#                     hnn_orbit, settings = get_sgp4_orbit(state, t_points=t_points, t_span=t_span, update_fn=update_fn)
#                     hnn_orbit_pos = hnn_orbit[0].T[:, [1, 2, 3]]
                    
#                     next_true_pos = trajectory_states[:, [0, 2, 4]]
#                     #print('Test Next True Pos: ', next_true_pos.shape)
#                     next_predicted_pos = torch.tensor(hnn_orbit_pos).to(device)
#                     #print('Test Next Predicted Pos: ', next_predicted_pos.shape)
                    
# #                     if args.verbose and step % args.print_every == 0:
# #                         print('\nExample Training Ground Truth: ', next_true_pos)
# #                         print('\nExample Training Prediction: ', next_predicted_pos)
                  
#                     test_loss_pos = L2_loss(next_true_pos, next_predicted_pos)
#                     #print('Test Loss Pos: ', test_loss_pos.item())
                    
                    # test step for position derivatives
                    
                    test_dxdt_hat = model.time_derivative(test_x[test_trajectory_start:test_trajectory_end, :])
                    test_dxdt_hat += input_noise * torch.randn(*test_x[test_trajectory_start:test_trajectory_end, :].shape).to(device) # add noise, maybe
                    
        #             if args.verbose and step % args.print_every == 0:
        #                 print('\nExample Testing Ground Truth: ', test_dxdt[test_ixs][0])
        #                 print('\nExample Testing Prediction: ', test_dxdt_hat[0])

                    test_loss_dxdt = L2_loss(test_dxdt[test_trajectory_start:test_trajectory_end, :], test_dxdt_hat)
                    #print('Test Loss dxdt: ', test_loss_dxdt.item())
            
                    #test_loss = test_loss_pos + test_loss_dxdt
                    test_loss = test_loss_dxdt
                    #print('Test Loss: ', test_loss.item())
                    
                    if math.isnan(test_loss.item()) or math.isnan(loss.item()):
                        print(trajectory_no)
                        print('lengths: ', len(lengths))
                        print(train_lengths_num)
                        print(test_lengths_num)
                        print(x.shape)
                        print(test_x.shape)
                        print(train_trajectory_start)
                        print(train_trajectory_end)
                        sys.exit()

                    # logging
                    stats['train_loss'].append(loss.item())
                    stats['test_loss'].append(test_loss.item())

                    tb.add_scalar('Train loss vs Steps', loss.item(), step)
                    tb.add_scalar('Test loss vs Steps', test_loss.item(), step)

                    if args.verbose and step % args.print_every == 0:
                      print("\nepoch {}, step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
                          .format(epoch_no, step, loss.item(), test_loss.item(), grad@grad, grad.std()))
                    if args.save_best_weights and step % args.print_every == 0:
                        if test_loss.item() < best_test_loss:
                            path = '{}/{}-satellite-orbits-{}-step={}-trial={}.tar'.format(save_dir, args.name, label, step, trial_no)
                            torch.save(model.state_dict(), path)
                            best_test_loss = test_loss.item()
                            
                    train_trajectory_start = train_trajectory_end + 1
                    step = step + 1

#               train_dxdt_hat = model.time_derivative(x)
#               train_dist = (dxdt - train_dxdt_hat)**2
#               test_dxdt_hat = model.time_derivative(test_x)
#               test_dist = (test_dxdt - test_dxdt_hat)**2
              print('\nBest test loss {:.4e}\n'.format(best_test_loss))
            
  return model, stats

if __name__ == "__main__":
    args = get_args()
    
    if args.baseline:
        tb = SummaryWriter(comment=' Model = <Baseline>, batch_size = ' + str(args.batch_size) +
                            ', learn_rate = ' + str(args.learn_rate) + 
                            ', hidden_dim = ' + str(args.hidden_dim) +
                            ', input_noise = ' + str(args.input_noise) + 
                            ', total_steps = ' + str(args.total_steps) + 
                            ', data_percentage_usage = ' + str(args.data_percentage_usage))
    else:
        tb = SummaryWriter(comment=' Model = <HNN>, batch_size = ' + str(args.batch_size) +
                            ', learn_rate = ' + str(args.learn_rate) + 
                            ', hidden_dim = ' + str(args.hidden_dim) +
                            ', input_noise = ' + str(args.input_noise) + 
                            ', total_steps = ' + str(args.total_steps) + 
                            ', data_percentage_usage = ' + str(args.data_percentage_usage))      
    
    os.makedirs(args.exp_dir) if not os.path.exists(args.exp_dir) else None
    save_dir = args.exp_dir + '/pretrained_models'
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    
    label = 'baseline' if args.baseline else 'hnn'
    model, stats = train(args, save_dir, tb, label)

    # save
    if args.satellite_problem:
        path = '{}/{}-satellite-orbits-{}-step-last.tar'.format(save_dir, args.name, label)
    else:
        path = '{}/{}-orbits-{}-step-last.tar'.format(save_dir, args.name, label)
    torch.save(model.state_dict(), path)
