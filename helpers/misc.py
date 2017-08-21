import sys
import subprocess
import uuid
import tensorflow as tf

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stderr.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def extend_options(parser):
    (options, args) = parser.parse_args()
    experiment_name = '{}_{}'.format(get_git_revision_short_hash(), str(uuid.uuid4())[:4])
    experiment_location = '{}/{}/'.format(options.logdir, options.repo_name)
    run_name = '{}/{}'.format(experiment_location, experiment_name)

    parser.add_option("--experiment_name",      dest="experiment_name",       default='{}'.format(experiment_name),     type='string')
    parser.add_option("--experiment_location",  dest="experiment_location",   default='{}'.format(experiment_location), type='string')
    parser.add_option("--run_name",             dest="run_name",              default='{}'.format(run_name),            type='string')
    (options, args) = parser.parse_args()
    return options

def check_tf_version():
    if tf.__version__ < '1.0.0':
        print('Needs TensorFlow 1.0.0 or more recent, found {}'.format(tf.__version__))
        sys.exit(1)
