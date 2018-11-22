# import utilities
import gym
import argparse
import shepherd_gym

# import stable-baselines utilities
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv

def main():
    parser = argparse.ArgumentParser(description='PPO baseline implementation')
    parser.add_argument('-e','--experiment',type=str,default='ppo_test',help='name of experiment')
    parser.add_argument('-w','--env',type=str,default='Shepherd-v0',help='name of gym environment')
    parser.add_argument('-m','--mode',type=str,default='train',help='mode to run experiment')
    parser.add_argument('-p','--policy',type=str,default='mlp',help='type of policy network')
    parser.add_argument('-t','--timesteps',type=int,default=10000,help='number of timesteps to train')
    parser.add_argument('-d','--datapath',type=str,default='../data',help='path to save results')
    args = parser.parse_args()

    mode = args.mode
    env_name = args.env
    policy = args.policy
    data_path = args.datapath
    timesteps = args.timesteps
    experiment = args.experiment

    exp_path = f'{data_path}/{experiment}'
    log_path = f'{exp_path}/log_{timesteps}'
    model_path = f'{exp_path}/model_{timesteps}'

    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    if policy=='mlp':
        policy_type = MlpPolicy
    else:
        policy_type = MlpLstmPolicy

    model = PPO2(policy_type, env, verbose=1, tensorboard_log=log_path)

    if mode == 'train':
        model.learn(total_timesteps=timesteps)
        model.save(model_path)
    else:
        model.load(model_path)

    env.render()
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, _, _, _ = env.step(action)

if __name__=='__main__':
    main()