
import argparse
from arm.main import main as main_arm
from flow_based.main import main as main_flow
from gan.main import main as main_gan
from diffusion.main import main as main_diff
from vae.main import main as main_vae
import sys

# from auto.main import main as main_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose the model to run')
    parser.add_argument('model', choices=['arm', 'flow', 'gan', 'diffusion', 'vae'], help='Select the model: arm or flow or gan or diffusion or vae')
    known_args, rest_args = parser.parse_known_args()

    if known_args.model == 'arm':
        sys.argv[1:] = rest_args 
        main_arm()
    elif known_args.model == 'flow':
        sys.argv[1:] = rest_args 
        main_flow()
    elif known_args.model == 'gan':
        sys.argv[1:] = rest_args 
        main_gan()
    elif known_args.model == 'diffusion':
        sys.argv[1:] = rest_args 
        main_diff()
    elif known_args.model == 'vae':
        sys.argv[1:] = rest_args 
        main_vae()