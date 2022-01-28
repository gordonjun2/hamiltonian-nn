# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data import get_dataset
from utils import L2_loss, to_pickle, from_pickle, get_model_parm_nums

def get_args():
    parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('--input_dim', default=2*4, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')   # original default is 200
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')          # original default is 1e-3
    parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
    parser.add_argument('--input_noise', default=0.0, type=int, help='std of noise added to inputs')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='2body', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn (solenoidal or conservative)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu_enable', dest='gpu_enable', action='store_true', help='include if gpu is to be used')
    parser.add_argument('--gpu_select', default=0, type=int, help='select which gpu to use')
    parser.add_argument('--satellite_problem', dest='satellite_problem', action='store_true', help='set scenario to be Satellite Problem instead of Two-Body Problem as demonstrated in the paper')
    parser.add_argument('--data_percentage_usage', default=1, type=float, help='percentage of data to use (eg. 1 means all data)')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  device = torch.device('cuda:' + str(args.gpu_select) if args.gpu_enable else 'cpu')

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  if args.satellite_problem:
    output_dim = 2*6 if args.baseline else 2
    nn_model = MLP(2*6, args.hidden_dim, output_dim, args.nonlinearity).to(device)
    model = HNN(2*6, differentiable_model=nn_model,
            field_type=args.field_type, baseline=args.baseline, device = device)
  else:
    output_dim = 2*4 if args.baseline else 2
    nn_model = MLP(2*4, args.hidden_dim, output_dim, args.nonlinearity).to(device)
    model = HNN(2*4, differentiable_model=nn_model,
            field_type=args.field_type, baseline=args.baseline, device = device)

  num_parm = get_model_parm_nums(model)
  
  print('\n')
  print('model contains {} parameters'.format(num_parm))
  print('\n')

  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)

  # arrange data
  data = get_dataset(args.name, args.save_dir, satellite_problem = args.satellite_problem, data_percentage_usage = args.data_percentage_usage, verbose=True)

  x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32).to(device)
  # Each line of 'x' is in the form (qx1, qx2, qy1, qy2, px1, px2, py1, py2) in original 2-body experiment and (qx1, qx2, qy1, qy2, qz1, qz2, px1, px2, py1, py2, pz1, pz2) in the satellite-problem experiment
    
  test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32).to(device)
  dxdt = torch.Tensor(data['dcoords']).to(device)
  test_dxdt = torch.Tensor(data['test_dcoords']).to(device)

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # train step
    
    # 'torch.randperm(x.shape[0])' randomizes array index (shuffling) and '[:args.batch_size]' slices the first 'batch_size' array index for training
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    dxdt_hat = model.time_derivative(x[ixs])
    dxdt_hat += args.input_noise * torch.randn(*x[ixs].shape).to(device) # add noise, maybe
    if args.verbose and step % args.print_every == 0:
        print('\nExample Training Ground Truth: ', dxdt[ixs][0])
        print('\nExample Training Prediction: ', dxdt_hat[0])
    loss = L2_loss(dxdt[ixs], dxdt_hat)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    optim.step() ; optim.zero_grad()

    # run test data
    test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    test_dxdt_hat = model.time_derivative(test_x[test_ixs])
    test_dxdt_hat += args.input_noise * torch.randn(*test_x[test_ixs].shape).to(device) # add noise, maybe
    if args.verbose and step % args.print_every == 0:
        print('\nExample Testing Ground Truth: ', test_dxdt[test_ixs][0])
        print('\nExample Testing Prediction: ', test_dxdt_hat[0])
    test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("\nstep {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
          .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))

  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('\nFinal train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}\n'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'hnn'
    if args.satellite_problem:
        path = '{}/{}-satellite-orbits-{}.tar'.format(args.save_dir, args.name, label)
    else:
        path = '{}/{}-orbits-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
