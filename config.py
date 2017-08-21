from optparse import OptionParser
import os

def set_config():
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 0.1")

    parser.add_option("--repo_name",            default='generative-models',
                            type='string',      help="Repo name")
    parser.add_option("--autoencoder_path",   default='/share/generative-models/autoencoder_path/model',
                            type='string',      help="Log, saved models and tensorboard for the autoencoder")
    return parser

if __name__ == '__main__':
    parser = set_config()
    parser.print_help()
