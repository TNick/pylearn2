"""Module containing the MultiSeq class and support functionality.

This class along with yaml extensions implemented in yaml_parse
(!multiseq and !range) allow you to define an experiment once while also
using a range of values for some of the parameters.

The simplest usage would imply defining both the control class (MultiSeq)
and the experiment in the same file like so:

.. code-block:: none

    {
        multiseq: !multiseq: {
            # ...
        },
        train: !obj:pylearn2.train.Train {
            # ...
        }
    }

"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Universite de Montreal"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import re
import os
from numpy import linspace
from random import randint
from datetime import datetime
from uuid import uuid1

from pylearn2.utils import as_floatX, string_utils
from pylearn2.utils.string_utils import preprocess

# Regular expression used to extract data from !range
RANGE_REGEXP = r'^\$\{([0-9a-zA-Z_]+):([0-9\.]+),([0-9\.]+),([0-9]+)\}$'


class RangeVar(object):
    """
    One range variable.

    Parameters
    ----------

    name : str
        The name as it will be used in the custome environment.
    space : numpy.linspace instance
        The linear space from which we will extract values..
    index : int
        Current index inside space.

    """
    def __init__(self, name, space):
        super(RangeVar, self).__init__()

        self.name = name
        self.space = space
        self.index = -1

    def compute_next_id(self, method):
        """
        Selects the next index based on current value and provided settings.

        If ordered mode is selected the values will be incremented and,
        at the end, will wrap around.MultiSeq

        If random mode is selected a random index is picked up.

        Returns
        -------
        new_idx : int
            Newly assigned index.
        """
        new_idx = self.index
        if method == 'ordered':
            new_idx = new_idx + 1
            if new_idx >= len(self.space):
                new_idx = 0
        elif method == 'random':
            new_idx = randint(0, len(self.space)-1)
        else:
            raise NotImplementedError(method)

        self.index = new_idx
        return new_idx

    def value(self):
        """
        Returns the value based on internal range and index.

        We can't just `float(self.space[self.index])` because it may get
        promoted to float64 and that is only desirable if floatX is float64.

        We can't also just `as_floatX(self.space[self.index])` because there
        are assertions in code that explicitly expect a float and
        as_floatX will create a one dimensional array.

        So, there you go. The solution below appears to work. I'm sure you
        can see why.
        """
        return float(as_floatX(self.space[self.index]))

class MultiSeq(object):
    """
    Implements multiple experiments from a single source using value ranges.

    This is a helper class for yaml_parser module.

    Parameters
    ----------

    multiseq_tag: str
        Unique tag for this run generated using uuid or current time.

    iterations: int
        Number of iterations performed so far.

    dynamic_env: dict of strings
        Variables that are updated at the start of a new iteration

    range_vars: list
        the list of variables with an associated range

    """
    _instance = None
    def __init__(self):

        super(MultiSeq, self).__init__()
        MultiSeq._instance = self

        self.multiseq_tag = None
        self.grouping = 'once'
        self.iterations = 0
        self.dynamic_env = {}
        self.range_vars = {}
        self.__range_regexp__ = re.compile(RANGE_REGEXP)


    @staticmethod
    def get_instance():
        """
        Get existing instance or create a new one.
        """
        if MultiSeq._instance is None:
            MultiSeq()
        return MultiSeq._instance

    @staticmethod
    def get_value(rng_var):
        """
        Get existing instance or create a new one.
        """
        return rng_var.value()

    def reinit(self, grouping, gen_val, gen_tag, max_count, environ=None):
        """
        Prepare to iterate over examples.
        """
        self.iterations = 0
        self.grouping = grouping
        self.gen_tag = gen_tag
        self.gen_val = gen_val
        self.max_count = max_count
        if environ is None:
            self.dynamic_env = {}
        else:
            self.dynamic_env = environ
        self.__order_of_change__ = []
        self.__var_to_change__ = -1

    def unique_tag(self):
        """
        Generate a unique tag for current run.
        """
        if self.gen_tag == 'uuid':
            self.multiseq_tag = str(uuid1())
        else:
            date_tm = datetime.utcnow()
            self.multiseq_tag = '%4d_%02d_%02d_%02d_%02d_%02d_%06d' % \
                                (date_tm.year, date_tm.month, date_tm.day,
                                 date_tm.hour, date_tm.minute, date_tm.second,
                                 date_tm.microsecond)

    def first_iteration(self):
        """
        Prepares the object for first run.
        """
        # prepare the counter
        self.iterations = 0

        # generate a unique tag
        self.unique_tag()

        # update all variables to next value
        for dynk in self.range_vars:
            self.range_vars[dynk].compute_next_id(self.gen_val)

        # we will change variables in this order
        self.__var_to_change__ = -1
        self.__order_of_change__ = self.range_vars.keys()

        # update variables in custom dictionary
        self.set_iter_vars()

    def next_iteration(self):
        """
        Advances to next step.

        Returns False if we're at the end.
        """
        # update the counter
        self.iterations = self.iterations + 1
        if self.iterations >= self.max_count:
            return False

        # generate a unique tag
        self.unique_tag()

        if self.grouping == 'once':
            return False
        elif self.grouping == 'one':
            # select the variable that is going to be changed
            if len(self.__order_of_change__) == 0:
                return False
            self.__var_to_change__ = self.__var_to_change__ + 1
            if self.__var_to_change__ >= len(self.__order_of_change__):
                self.__var_to_change__ = 0

            # only update this variable to next value
            var_name = self.__order_of_change__[self.__var_to_change__]
            self.range_vars[var_name].compute_next_id(self.gen_val)
        else:
            # update all variables to next value
            for dynk in self.range_vars:
                self.range_vars[dynk].compute_next_id(self.gen_val)

        # update variables in custom dictionary
        self.set_iter_vars()

        # continue the iteration
        return True

    def set_iter_vars(self):
        """
        Set the variables that change with each iteration.
        """
        self.dynamic_env['MULTISEQ_ITER'] = str(self.iterations)
        self.dynamic_env['MULTISEQ_TAG'] = self.multiseq_tag

        # add our range variables to custom environment
        for dynk in self.range_vars:
            self.dynamic_env[dynk] = str(self.val_for_name(dynk))

    @staticmethod
    def _to_number(str_to_conv):
        """
        Attempts to convert the value to either an integer or a float.
        """
        try:
            return int(str_to_conv)
        except ValueError:
            return float(str_to_conv)

    def add_range_var(self, var_str):
        """
        Creates or updated a range variable from a string.

        Expected string has the form "${VARNAME:start,stop,num}"
        """
        match = self.__range_regexp__.match(var_str)
        if not match:
            raise ValueError('Value %s is not a valid !range value' % str(var_str))

        start_val = MultiSeq._to_number(match.group(2))
        end_val = MultiSeq._to_number(match.group(3))
        num_val = MultiSeq._to_number(match.group(4))
        lspc = linspace(start=start_val,
                        stop=end_val,
                        num=num_val,
                        endpoint=True)

        # create the variable
        rng_var = RangeVar(name=match.group(1), space=lspc)

        # insert it in our list
        self.range_vars[match.group(1)] = rng_var

        # and return initial value
        return rng_var

    def val_for_name(self, rng_var_name):
        """
        Gets current value of a range variable given range variable's name.
        """
        return self.range_vars[rng_var_name].value()

    def save_params(self):
        """
        Converts the input to a string and dumps it to provided file.

        The file is only generated if local environment contains
        a key called `PARAMETERS_FILE`. Otherwise we assume that the
        user does not want a separate file.

        TODO:generate a runnable yaml file.
        """
        if not self.dynamic_env.has_key('PARAMETERS_FILE'):
            return
        file_path = preprocess(self.dynamic_env['PARAMETERS_FILE'],
                               self.dynamic_env)
        with open(file_path, "w") as f_param:
            for k in self.dynamic_env.keys():
                f_param.write('%s: %s\n' % \
                              (k,
                               preprocess(self.dynamic_env[k],
                                          environ=self.dynamic_env)))
            f_param.write('#----------------------------------------\n')
            for k in os.environ.keys():
                f_param.write('%s: %s\n' % (k, os.environ[k]))
