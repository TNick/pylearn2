#!/usr/bin/env python
"""
Script implementing the logic for training pylearn2 models.

This is a "driver" that is capable of dealing with yaml files that
use range of values as parameters.

Basic usage:

.. code-block:: none

    train_multi.py yaml_file.yaml

The YAML file should contain a pylearn2 YAML description of a
dictionary with two entries: one that describes common characteristics for
all the experiments that will be performed and one that describes the
experiments themselves as a `pylearn2.train.Train` object
(or optionally, a list of Train objects to run sequentially).

.. code-block:: none

    {
        multiseq: !multiseq: {
            # ...
        },
        train: !obj:pylearn2.train.Train {
            # ...
        }
    }

.. code-block:: none

    {
        multiseq: !multiseq: {
            # ...
        },
        train: [
            !obj:pylearn2.train.Train {
                # ...
            },
            !obj:pylearn2.train.Train {
                # ...
            },
        ]
    }

See `doc/yaml_tutorial` for a description of how to write the YAML syntax.

The following environment variables will be locally defined and available
for use within the YAML file:

- `PYLEARN2_TRAIN_BASE_NAME`: the name of the file within the directory
  (`foo/bar.yaml` -> `bar.yaml`)
- `PYLEARN2_TRAIN_DIR`: the directory containing the YAML file
  (`foo/bar.yaml` -> `foo`)
- `PYLEARN2_TRAIN_FILE_FULL_STEM`: the filepath with the file extension
  stripped off.
  `foo/bar.yaml` -> `foo/bar`)
- `PYLEARN2_TRAIN_FILE_STEM`: the stem of `PYLEARN2_TRAIN_BASE_NAME`
  (`foo/bar.yaml` -> `bar`)
- `PYLEARN2_TRAIN_PHASE` : set to `phase0`, `phase1`, etc. during iteration
  through a list of Train objects. Not defined for a single train object.

These environment variables are especially useful for setting the save
path. For example, to make sure that `foo/bar.yaml` saves to `foo/bar.pkl`,
use

.. code-block:: none

    save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}.pkl"

This way, if you copy `foo/bar.yaml` to `foo/bar2.yaml`, the output of
`foo/bar2.yaml` won't overwrite `foo/bar.pkl`, but will automatically save
to foo/bar2.pkl.

Use `train.py -h` to see an auto-generated description of advanced options.

TODO: train and train_multi share a lot of code; factor out common parts
in a module, then refactor.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Universite de Montreal"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
# Standard library imports
import argparse
import gc
import logging
import os
from time import strftime

# Local imports
from pylearn2.utils import serial, string_utils
from pylearn2.utils.logger import (
    CustomStreamHandler, CustomFormatter, restore_defaults
)
from pylearn2.config import yaml_parse

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Launch an experiment from a YAML configuration file.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--level-name', '-L',
                        action='store_true',
                        help='Display the log level (e.g. DEBUG, INFO) '
                             'for each logged message')
    parser.add_argument('--timestamp', '-T',
                        action='store_true',
                        help='Display human-readable timestamps for '
                             'each logged message')
    parser.add_argument('--time-budget', '-t', type=int,
                        help='Time budget in seconds. Stop training at '
                             'the end of an epoch if more than this '
                             'number of seconds has elapsed.')
    parser.add_argument('--verbose-logging', '-V',
                        action='store_true',
                        help='Display timestamp, log level and source '
                             'logger for every logged message '
                             '(implies -T).')
    parser.add_argument('--debug', '-D',
                        action='store_true',
                        help='Display any DEBUG-level log messages, '
                             'suppressed by default.')
    parser.add_argument('--skipexc',
                        action='store_true',
                        help='Skip exceptions in  main loop.')
    parser.add_argument('config', action='store',
                        choices=None,
                        help='A YAML configuration file specifying the '
                             'training procedure')
    return parser


def _getVar(key, environ=None):
    """
    Looks for a key in custom and os environments.

    Parameters
    ----------
    key : str
        The key to look for.
    environ : dict, optional
        A custom dictionary to search befor system environment.

    Returns
    -------
        None if the key was not found, a string otherwise.
    """
    if environ:
        if environ.has_key(key):
            return string_utils.preprocess(environ[key], environ=environ)
    if os.environ.has_key():
        return string_utils.preprocess(os.environ[key])
    return None

def _getBestResult(experiment):
    """
    If the experiment uses MonitorBasedSaveBest wi;; extract last best result.
    """
    try:
        return ' %12.8f' % \
               experiment.model.tag['MonitorBasedSaveBest']['best_cost'].item(0)
    except:
        return ' '*13


def train(config, level_name=None, timestamp=None, time_budget=None,
          verbose_logging=None, debug=None, environ=None, skip_exceptions=False):
    """
    Trains a given YAML file.

    Parameters
    ----------
    config : str
        A YAML configuration file specifying the
        training procedure.
    level_name : bool, optional
        Display the log level (e.g. DEBUG, INFO)
        for each logged message.
    timestamp : bool, optional
        Display human-readable timestamps for
        each logged message.
    time_budget : int, optional
        Time budget in seconds. Stop training at
        the end of an epoch if more than this
        number of seconds has elapsed.
    verbose_logging : bool, optional
        Display timestamp, log level and source
        logger for every logged message
        (implies timestamp and level_name are True).
    debug : bool, optional
        Display any DEBUG-level log messages,
        False by default.
    environ : dict, optional
        Custom variables to be replaced in yaml file.
    """

    # Undo our custom logging setup.
    restore_defaults()
    # Set up the root logger with a custom handler that logs stdout for INFO
    # and DEBUG and stderr for WARNING, ERROR, CRITICAL.
    root_logger = logging.getLogger()
    if verbose_logging:
        formatter = logging.Formatter(fmt="%(asctime)s %(name)s %(levelname)s "
                                          "%(message)s")
        handler = CustomStreamHandler(formatter=formatter)
    else:
        if timestamp:
            prefix = '%(asctime)s '
        else:
            prefix = ''

        if level_name:
            prefix = prefix + '%(levelname)s '

        formatter = CustomFormatter(prefix=prefix, only_from='pylearn2')
        handler = CustomStreamHandler(formatter=formatter)
    root_logger.addHandler(handler)

    # Set the root logger level.
    if debug:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)


    # publish environment variables relevant to this file
    serial.prepare_train_file(config)

    # load the tree of Proxy objects
    yaml_tree = yaml_parse.load_path(config,
                                     instantiate=False,
                                     environ=environ)

    if not isinstance(yaml_tree, dict):
        raise ValueError('.yaml file is expected to contain a dictionary as the root object')
    elif not (yaml_tree.has_key('multiseq') and yaml_tree.has_key('train')):
        raise ValueError('.yaml file is expected to have two keys: multiseq and train')

    # the two important objects
    multiseq = yaml_tree['multiseq']
    train_list = yaml_tree['train']
    if not isinstance(train_list, list):
        train_list = [train_list]

    # prepare to run
    multiseq.first_iteration()
    cont_flag = True

    # see if we're going to generate a report
    repot_f = None
    report = _getVar('PYLEARN2_REPORT', multiseq.dynamic_env)
    if report:
        repot_f = open(report, "a")
        hdr = (' Index      '
               'Date & Time                       '
               'Tag                  '
               'Finish    Train    Total Batchs  Epchs  Exampls   '
               'Best result  Tests in file\n')
        hdr_guards = '-' * len(hdr) + '\n'
        repot_f.write('\n')
        repot_f.write(hdr_guards)
        repot_f.write(hdr)
        repot_f.write(hdr_guards)
        repot_f.write('\n')

    emergency_exit = False
    while cont_flag:

        # log to user and start a new line for the report
        root_logger.debug('Run %s with tag %s',
                          multiseq.dynamic_env['MULTISEQ_ITER'],
                          multiseq.dynamic_env['MULTISEQ_TAG'])
        if repot_f:
            repot_f.write('%6s  %19s  %36s  ' % \
                         (multiseq.dynamic_env['MULTISEQ_ITER'],
                          strftime("%Y %m %d %H:%M:%S"),
                          multiseq.dynamic_env['MULTISEQ_TAG']))

        # update the environment with our dynamic variables
        # yaml_parse.additional_environ = multiseq.dynamic_env
        os.environ.update(multiseq.dynamic_env)

        # as this wil probably be an unattended process we may want to
        # tolerate exceptions
        try:

            # TODO: we are accesing a protected member here
            # either change the name or define a wrapper
            train_list_inst = yaml_parse._instantiate(train_list)

            # if the environment defines a PARAMETERS_FILE variable
            # dump the parameters there as simple text
            multiseq.save_params()

            # perform all the tests/experiments once
            first_subobj = True
            subobj_completed = 0
            for number, subobj in enumerate(train_list_inst):

                # Publish a variable indicating the training phase.
                phase_variable = 'PYLEARN2_TRAIN_PHASE'
                phase_value = 'phase%d' % (number + 1)
                os.environ[phase_variable] = phase_value

                # Execute this training phase.
                try:
                    subobj.main_loop(time_budget=time_budget)
                except:
                    subobj.tear_down()
                    raise

                # log first train to the report
                if first_subobj and repot_f:
                    repot_f.write('%6s %8d %8d %6d %6d %8d %s' % \
                              (str(subobj.model.monitor.training_succeeded),
                               subobj.training_seconds.get_value().item(0),
                               subobj.total_seconds.get_value().item(0),
                               subobj.model.monitor.get_batches_seen(),
                               subobj.model.monitor.get_epochs_seen(),
                               subobj.model.monitor.get_examples_seen(),
                               _getBestResult(subobj)))

                    # TODO: report performance here or channels
                    # for best model according to objective

                first_subobj = False

                # Clean up, in case there's a lot of memory used that's
                # necessary for the next phase.
                # TODO: because subobj is part of a bigger object it may be
                # that it does not get cleaned up here.
                del subobj
                gc.collect()
                subobj_completed = subobj_completed + 1

        except (KeyboardInterrupt, SystemExit):
            emergency_exit = True
            if repot_f:
                repot_f.write('%50s' % 'User terminated')

        except Exception, exc:
            if skip_exceptions:
                if repot_f:
                    repot_f.write('%50s' % str(exc))
            else:
                raise

        # we've completed a run; finalize report line for it
        if repot_f:
            repot_f.write(' %4d completed\n' % subobj_completed)
            repot_f.flush()

        # user requested exit
        if emergency_exit:
            break

        # LiveMonitoring seems to stay behind and binded to same port
        # That will throw an exception on next run
        # We try to mitigate that here.
        del train_list_inst
        gc.collect()

        # prepare next run

        cont_flag = multiseq.next_iteration()

    # done with the report
    if repot_f:
        repot_f.close()

def main():
    """
    Module entry point.
    """
    local_parser = make_argument_parser()
    args = local_parser.parse_args()
    train(config=args.config,
          level_name=args.level_name,
          timestamp=args.timestamp,
          time_budget=args.time_budget,
          verbose_logging=args.verbose_logging,
          debug=args.debug,
          environ=None,
          skip_exceptions=args.skipexc)

if __name__ == "__main__":
    # See module-level docstring for a description of the script.
    main()
