from spinup.utils.run_utils import ExperimentGrid
from spinup import tac
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()
    
    eg = ExperimentGrid(name='tac_exp')
    eg.add('env_name', args.env, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('q', [1.0,1.5,2.0,2.5,3.0,5.0,10.0])
    eg.run(tac, num_cpu=args.cpu)
