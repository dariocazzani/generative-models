from optparse import OptionParser
import os

def set_config():
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 0.1")

    parser.add_option("--MAIN_PATH",            default='{}'.format(os.getenv("MAIN_PATH")),
                            type=str,           help="Log, saved models and tensorboard for the autoencoder")
    parser.add_option("--DATA_PATH",            default='{}'.format(os.getenv("DATA_PATH")),
                            type=str,           help='Store Data')
    parser.add_option("--z_dim",                default=2,
                            type=int,           help="Latent Variable Dimensions")
    parser.add_option("--batch_size",           default=128,
                            type=int,           help="Set batch size")
    parser.add_option("--epochs",               default=1000,
                            type=int,           help="Set Number of Epochs")
    parser.add_option("--learning_rate",        default=1E-3,
                            type=float,         help="Set Starting Learning Rate")
    parser.add_option("--beta1",                default=0.9,
                            type=float,         help="Set Beta 1 value for AdamOptimizer")
    parser.add_option("--run_inference",        default=False,
                            action="store_true",help="Train or just run inference to generate image")
    parser.add_option("--save_plots",           default=False,
                            action="store_true",help="Save plots in folder out every 50 iterations")

    return parser

if __name__ == '__main__':
    parser = set_config()
    parser.print_help()
