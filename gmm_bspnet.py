import numpy as np
import sklearn.mixture
import torch
import h5py
from pathlib import Path
import os

from gmm_torch.gmm import GaussianMixture


def get_all_zcodes_for_class(classid):
    hdf5_path = 'checkpoint/all_vox256_img_ae_64/all_vox256_img_train_z.hdf5'
    hdf5_file = h5py.File(hdf5_path, mode='r')
    z_key = list(hdf5_file.keys())[0]
    z_codes_all = torch.from_numpy(hdf5_file[z_key][:]).float()
    hdf5_file.close()
    sample_names = Path('data/all_vox256_img/all_vox256_img_train.txt').read_text().splitlines()
    valid_indices = [i for i, name in enumerate(sample_names) if name.startswith(classid)]

    z_codes = z_codes_all[valid_indices, :]
    return z_codes


def gmmfit_class(classid):
    n_components = 32
    dim = 256

    z_codes = get_all_zcodes_for_class(classid)

    model = GaussianMixture(n_components, dim, covariance_type='diag')
    model.fit(z_codes)
    return model


def sample_gmm(classid, num_samples=1):
    model = gmmfit_class(classid)
    z = model.sample(num_samples)[0]
    return z


def sample_train(classid, num_samples=1):
    z_codes_all = get_all_zcodes_for_class(classid)
    indices = np.random.choice(z_codes_all.shape[0], num_samples, replace=False)
    z = z_codes_all[indices, :]
    return z


def get_bsp():
    from modelAE import BSP_AE
    from modelSVR import BSP_SVR

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", action="store", dest="phase", default=1, type=int,
                        help="phase 0 = continuous, phase 1 = hard discrete, phase 2 = hard discrete with L_overlap, phase 3 = soft discrete, phase 4 = soft discrete with L_overlap [1]")
    parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
    parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.0001, type=float, help="Learning rate for adam [0.0001]")
    parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
    parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
    parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
    parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/", help="Root directory of dataset [data]")
    parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
    parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
    parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
    parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
    parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
    parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
    parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
    parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
    parser.add_argument("--evalz", action="store_true", dest="evalz", default=False, help="True for getting latent codes [False]")
    parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="Which GPU to use [0]")
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    bsp_ae = BSP_AE(FLAGS)
    return bsp_ae, FLAGS


def main():
    z = sample_gmm('03001627', 20)
    # z = sample_train('03001627', 20)
    z = z.cuda()
    bsp, config = get_bsp()
    bsp.eval_z(config, z)


if __name__ == "__main__":
    main()
