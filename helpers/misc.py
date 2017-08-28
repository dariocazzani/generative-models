import sys
import subprocess
import uuid
import tensorflow as tf
import os

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stderr.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def extend_options(parser, project_name, script_name):
    (options, args) = parser.parse_args()

    project_folder = '{}/{}'.format(options.MAIN_PATH, project_name)
    experiment_name = '{}'.format(str(uuid.uuid4().hex)[4:])
    experiment_folder = '{}/{}_{}'.format(project_folder, script_name, experiment_name)

    parser.add_option("--project_folder",       dest="project_folder",
                        default='{}'.format(project_folder),                type='string')

    parser.add_option("--experiment_name",      dest="experiment_name",
                        default='{}'.format(experiment_name),               type='string')

    parser.add_option("--experiment_folder",    dest="experiment_folder",
                        default=experiment_folder, type='string')

    parser.add_option("--checkpoints_path",     dest="checkpoints_path",
                        default='{}/checkpoints/'.format(experiment_folder),   type='string')

    parser.add_option("--tensorboard_path",     dest="tensorboard_path",
                        default='{}/tensorboard'.format(experiment_folder),    type='string')

    parser.add_option("--logs_path",            dest="logs_path",
                        default='{}/logs'.format(experiment_folder),           type='string')

    (options, args) = parser.parse_args()

    # Make sure that folders and file exits logs.txt exists
    if not os.path.exists(experiment_folder):
        subprocess.call(['mkdir','{}'.format(options.project_folder)])
        subprocess.call(['mkdir','{}'.format(options.experiment_folder)])
        subprocess.call(['mkdir','{}'.format(options.tensorboard_path)])
        subprocess.call(['mkdir','{}'.format(options.checkpoints_path)])
        subprocess.call(['mkdir','{}'.format(options.logs_path)])
        subprocess.call(['touch','{}/log.txt'.format(options.logs_path)])

    return options

def check_tf_version():
    if tf.__version__ < '1.0.0':
        print('Needs TensorFlow 1.0.0 or more recent, found {}'.format(tf.__version__))
        sys.exit(1)
