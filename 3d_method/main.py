import os
import time
import argparse
from utils import *
from cyclegan3d import CycleGAN
from dataloader import CreateDataloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

def find_latest_weights(pretrained_path):
    files = os.listdir(pretrained_path)
    g_a2b_weights = [f for f in files if 'G_A2B.h5' in f]
    g_b2a_weights = [f for f in files if 'G_B2A.h5' in f]
    d_a_weights = [f for f in files if 'D_A.h5' in f]
    d_b_weights = [f for f in files if 'D_B.h5' in f]

    if not (g_a2b_weights and g_b2a_weights and d_a_weights and d_b_weights):
        raise FileNotFoundError("One or more weight files are missing in the pretrained path")

    latest_g_a2b = max(g_a2b_weights, key=lambda x: int(x.split('_')[0]))
    latest_g_b2a = max(g_b2a_weights, key=lambda x: int(x.split('_')[0]))
    latest_d_a = max(d_a_weights, key=lambda x: int(x.split('_')[0]))
    latest_d_b = max(d_b_weights, key=lambda x: int(x.split('_')[0]))

    return latest_g_a2b, latest_g_b2a, latest_d_a, latest_d_b

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, required=True, help='location where the data is stored')
parser.add_argument('--out_path', type=str, required=True, help='location where to save results')
parser.add_argument('--max_iterations', default=1000, type=int, nargs='?', help='select the total number of iterations for training')
parser.add_argument('--resume_training', default=False, type=bool, nargs='?', help='if resuming cycleGAN training. It also requires pretrained weights path')
parser.add_argument('--pretrained_path', default='', type=str, nargs='?', help='pretrained weights path for loading the generator and discriminator')
parser.add_argument('--save_train_freq', default=100, type=int, nargs='?', help='frequency to save training images')
parser.add_argument('--save_weights_freq', default=200, type=int, nargs='?', help='frequency to save generator and discriminator weights')
parser.add_argument('--batch_size', default=1, nargs='?', const=1,  type=str, help='batch size for training and testing')
parser.add_argument('--g_residual_blocks', default=9, type=str, nargs='?', help='the number of residual blocks in the generator bottleneck')
parser.add_argument('--lr_G', default=0.0002, nargs='?', const=1, help='generator learning rate')
parser.add_argument('--lr_D', default=0.0002, nargs='?', const=1, help='discriminator learning rate')

args = parser.parse_args()

def main():
    print("[INFO] CycleGAN training initiated ...")
    train_data_loader = CreateDataloader(args, mode='train', shuffle=True, cache=True)
    train_data_num = len(train_data_loader)

    date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    out_path = os.path.join(args.out_path, date_time)

    if not os.path.exists(os.path.join(out_path, 'images')):
        os.makedirs(os.path.join(out_path, 'images'))
        os.makedirs(os.path.join(out_path, 'saved_weights'))

    print(f'[INFO] the results will be saved to {date_time} directory ...')

    gan = CycleGAN(args)

    if args.resume_training and args.pretrained_path:
        latest_g_a2b, latest_g_b2a, latest_d_a, latest_d_b = find_latest_weights(args.pretrained_path)
        gan.G_A2B.load_weights(os.path.join(args.pretrained_path, latest_g_a2b))
        gan.G_B2A.load_weights(os.path.join(args.pretrained_path, latest_g_b2a))
        gan.D_A.load_weights(os.path.join(args.pretrained_path, latest_d_a))
        gan.D_B.load_weights(os.path.join(args.pretrained_path, latest_d_b))
        print("[INFO] Loaded pretrained weights for resuming training")

    D_A_losses, D_B_losses, G_A2B_losses, G_B2A_losses, cycle_A_losses, cycle_B_losses = ([] for i in range(6))

    iteration = 0
    while iteration < args.max_iterations:
        for loop_index, batch_data in enumerate(train_data_loader):
            imgA, imgB = batch_data["imgA"].detach().cpu().numpy()[0,...,None], batch_data["imgB"][0,...,None].detach().cpu().numpy()

            fake_A, fake_B, cycle_A, cycle_B, G_A2B_loss, G_B2A_loss, cycle_A_loss, cycle_B_loss, D_A_loss, D_B_loss = gan.train_step(imgA, imgB)
        
            pred_slice = 10
            if iteration % args.save_train_freq == 0:
                save_tmp_images(iteration, imgA[:,...,pred_slice,:], imgB[:,...,pred_slice,:], 
                               fake_A[:,...,pred_slice,:], fake_B[:,...,pred_slice,:],
                               cycle_A[:,...,pred_slice,:], cycle_B[:,...,pred_slice,:], 
                               out_path)

            print(f'Iteration [{iteration}/{args.max_iterations}]', f'Loop index [{loop_index}/{train_data_num}]')

            D_A_losses.append(D_A_loss.numpy())
            D_B_losses.append(D_B_loss.numpy())
            G_A2B_losses.append(G_A2B_loss.numpy())
            G_B2A_losses.append(G_B2A_loss.numpy())
            cycle_A_losses.append(cycle_A_loss.numpy())
            cycle_B_losses.append(cycle_B_loss.numpy())

            if iteration % args.save_weights_freq == 0:
                gan.G_A2B.save_weights(f'{out_path}/saved_weights/{iteration}_G_A2B.h5')
                gan.G_B2A.save_weights(f'{out_path}/saved_weights/{iteration}_G_B2A.h5')
                gan.D_A.save_weights(f'{out_path}/saved_weights/{iteration}_D_A.h5')
                gan.D_B.save_weights(f'{out_path}/saved_weights/{iteration}_D_B.h5')

            iteration += 1

if __name__ == "__main__":
    main()

