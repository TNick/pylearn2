"""
Training extension for allowing querying of monitoring values while an
experiment executes.
"""
__authors__ = "Dustin Webb, Adam Stone, Nicu Tofan"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Dustin Webb", "Adam Stone"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import copy
import logging
LOG = logging.getLogger(__name__)

try:
    import zmq
    ZMQ_AVAILABLE = True
except Exception:
    ZMQ_AVAILABLE = False

try:
    from PySide import QtCore, QtGui

    import matplotlib
    import numpy as np
    matplotlib.use('Qt4Agg')
    matplotlib.rcParams['backend.qt4'] = 'PySide'

    from matplotlib.backends.backend_qt4agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar)
    from matplotlib.figure import Figure

    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PYPLOT_AVAILABLE = True
except ImportError:
    PYPLOT_AVAILABLE = False

MPLDC_AVAILABLE = False
try:
    if PYPLOT_AVAILABLE:
        import mpldatacursor as mpldc
        MPLDC_AVAILABLE = True
except ImportError:
    pass

from functools import wraps
from pylearn2.monitor import Monitor
from pylearn2.train_extensions import TrainExtension


class LiveMonitorMsg(object):
    """
    Base class that defines the required interface for all Live Monitor
    messages.
    """
    response_set = False

    def get_response(self):
        """
        Method that instantiates a response message for a given request
        message. It is not necessary to implement this function on response
        messages.
        """
        raise NotImplementedError('get_response is not implemented.')


class ChannelListResponse(LiveMonitorMsg):
    """
    A message containing the list of channels being monitored.
    """
    pass


class ChannelListRequest(LiveMonitorMsg):
    """
    A message indicating a request for a list of channels being monitored.
    """
    @wraps(LiveMonitorMsg.get_response)
    def get_response(self):
        return ChannelListResponse()


class ChannelsResponse(LiveMonitorMsg):
    """
    A message containing monitoring data related to the channels specified.
    Data can be requested for all epochs or select epochs.

    Parameters
    ----------
    channel_list : list
        A list of the channels for which data has been requested.

    start : int
        The starting epoch for which data should be returned.

    end : int
        The epoch after which data should be returned.

    step : int
        The number of epochs to be skipped between data points.
    """
    def __init__(self, channel_list, start, end, step=1):
        assert(
            isinstance(channel_list, list)
            and len(channel_list) > 0
        )
        self.channel_list = channel_list

        assert start >= 0
        self.start = start

        self.end = end

        assert step > 0
        self.step = step

        # This is the payload and it may be a Throwable object or
        # the actual result (exact type depends on the message type)
        self.data = {}

class ChannelsRequest(LiveMonitorMsg):
    """
    A message for requesting data related to the channels specified.

    Parameters
    ----------
    channel_list : list
        A list of the channels for which data has been requested.

    start : int
        The starting epoch for which data should be returned.

    end : int
        The epoch after which data should be returned.

    step : int
        The number of epochs to be skipped between data points.
    """
    def __init__(self, channel_list, start=0, end=-1, step=1):
        assert(
            isinstance(channel_list, list)
            and len(channel_list) > 0
        )
        self.channel_list = channel_list

        assert start >= 0
        self.start = start

        self.end = end

        assert step > 0
        self.step = step

    @wraps(LiveMonitorMsg.get_response)
    def get_response(self):
        return ChannelsResponse(
            self.channel_list,
            self.start,
            self.end,
            self.step
        )


class LiveMonitoring(TrainExtension):
    """
    A training extension for remotely monitoring and filtering the channels
    being monitored in real time. PyZMQ must be installed for this extension
    to work.

    A LiveMonitoring that has no connected subscribers will simply
    drop all messages.

    Parameters
    ----------
    address : string
        The IP addresses of the interfaces on which the monitor should listen.

    req_port : int
        The port number to be used to service request. To disable the
        request-response variant pass 0 to this parameter.

    pub_port : int
        The port number to be used to publish updates. To disable the
        publish-subscribe variant pass 0 to this parameter.
    """
    def __init__(self, address='*', req_port=5555, pub_port=5556):
        if not ZMQ_AVAILABLE:
            raise ImportError('zeromq needs to be installed to '
                              'use this module.')

        self.address = 'tcp://%s' % address

        assert req_port != pub_port

        assert req_port > 1024 and req_port < 65536
        self.req_port = req_port

        assert pub_port > 1024 and pub_port < 65536
        self.pub_port = pub_port

        address_template = self.address + ':%d'
        self.context = zmq.Context()

        self.req_sock = None
        if self.req_port > 0:
            self.req_sock = self.context.socket(zmq.REP)
            self.req_sock.bind(address_template % self.req_port)

        self.pub_sock = None
        if self.pub_port > 0:
            self.pub_sock = self.context.socket(zmq.PUB)
            self.pub_sock.bind(address_template % self.pub_port)

        # Tracks the number of times on_monitor has been called
        self.counter = 0

        # number of entries to be published at a time;
        # will be initialized the first time a message is about to be
        # published in __build_channel_resp__()
        self.post_size = 0

    def __build_channel_resp__(self, monitor, channel_list,
                               start=0, end=-1, step=1):
        """
        Constructs a response or publish message containing channel data.

        The message will either be an Throwable or a dictionary, with
        keys being the names of the channels.

        Individual entries for each channel may also be a Throwableinstance
        or channel data (see source for actual content).

        Parameters
        ----------
        monitor : Monitor
            Model's monitor from where we are about to extract the data

        channel_list : list
            A list of the channels for which data is needed.

        start : int
            The starting epoch for which data should be returned.

        end : int
            The epoch after which data should be returned.

        step : int
            The number of epochs to be skipped between data points.
        """
        result = {}
        if not isinstance(channel_list, list) or len(channel_list) == 0:
            channel_list = []
            result = TypeError('ChannelResponse requires a list of channels.')
        else:
            for channel_name in channel_list:
                if channel_name in monitor.channels.keys():
                    chan = copy.deepcopy(
                        monitor.channels[channel_name]
                    )
                    if self.post_size == 0:
                        self.post_size = len(chan.batch_record)
                    if end == -1:
                        end = len(chan.batch_record)
                    # TODO copying and truncating the records individually
                    # like this is brittle. Is there a more robust
                    # solution?
                    chan.batch_record = chan.batch_record[
                        start:end:step
                    ]
                    chan.epoch_record = chan.epoch_record[
                        start:end:step
                    ]
                    chan.example_record = chan.example_record[
                        start:end:step
                    ]
                    chan.time_record = chan.time_record[
                        start:end:step
                    ]
                    chan.val_record = chan.val_record[
                        start:end:step
                    ]
                    result[channel_name] = chan
                else:
                    result[channel_name] = KeyError(
                        'Invalid channel: %s' % channel_name
                    )
        return result

    def __reply_to_req__(self, monitor):
        """
        Replies to a request for specific channels or to list all channels.

        Parameters
        ----------
        monitor : Monitor
            Model's monitor from where we are about to extract the data

        """
        try:
            rsqt_msg = self.req_sock.recv_pyobj(flags=zmq.NOBLOCK)

            # Determine what type of message was received
            rsp_msg = rsqt_msg.get_response()

            if isinstance(rsp_msg, ChannelListResponse):
                rsp_msg.data = list(monitor.channels.keys())

            elif isinstance(rsp_msg, ChannelsResponse):
                channel_list = rsp_msg.channel_list
                rsp_msg.data = self.__build_channel_resp__(monitor,
                                                           channel_list,
                                                           rsp_msg.start,
                                                           rsp_msg.end,
                                                           rsp_msg.step)
            self.req_sock.send_pyobj(rsp_msg)
        except zmq.Again:
            pass

    def __publish_results__(self, monitor):
        """
        Publishes all channels to dedicated ZMQ slot.

        Parameters
        ----------
        monitor : Monitor
            Model's monitor from where we are about to extract the data
        """
        if self.pub_sock is None:
            return

        try:
            channel_list = list(monitor.channels.keys())
            start = self.counter*self.post_size
            end = -1 if self.post_size == 0 else start + self.post_size
            rsp_msg = ChannelsResponse(channel_list, start, end, step=1)
            rsp_msg.data = self.__build_channel_resp__(monitor,
                                                       channel_list,
                                                       start, end)
            self.pub_sock.send_pyobj(rsp_msg)
        except Exception, ex:
            LOG.warn("Exception while publishing results in LiveMonitoring:" +
                     ex.message)

    @wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        monitor = Monitor.get_monitor(model)
        if self.req_port > 0:
            self.__reply_to_req__(monitor)
        if self.pub_port:
            self.__publish_results__(monitor)
        self.counter += 1

class LiveMonitor(object):
    """
    A utility class for requested data from a LiveMonitoring training
    extension.

    On the publish-subscribe variant, please note that, if the subscriber
    is slower than the publisher, the messages will pile up on the publisher.

    Parameters
    ----------
    address : string
        The IP address on which a LiveMonitoring process is listening.

    req_port : int
        The port number on which a LiveMonitoring process is listening.

    subscribe : bool
        Use publish-subscribe variant (True) or request-reply (False, default).
    """
    def __init__(self, address='127.0.0.1', req_port=5555, subscribe=False):
        """
        """
        if not ZMQ_AVAILABLE:
            raise ImportError('zeromq needs to be installed to '
                              'use this module.')

        self.address = 'tcp://%s' % address

        assert req_port > 0
        self.req_port = req_port
        self.subscribe = subscribe

        self.context = zmq.Context()

        if subscribe:
            self.req_sock = self.context.socket(zmq.SUB)
            self.req_sock.setsockopt(zmq.SUBSCRIBE, "")
        else:
            self.req_sock = self.context.socket(zmq.REQ)

        self.req_sock.connect(self.address + ':' + str(self.req_port))

        # A dictionary that has the names of the channels as keys and
        # channel data reported by LiveMonitoring as values.
        self.channels = {}

    def list_channels(self, cached=False):
        """
        Returns a list of the channels being monitored.

        Parameters
        ----------
        cached : bool
            If a cached version exists, return that instead of sending a
            new request.
        """
        if cached:
            if len(self.channels) > 0:
                return self.channels.keys()

        if self.subscribe:
            # we could create a new socket and send a request here
            LOG.warn('Subscribe variant of LiveMonitor is only capable '
                     'of returning cached list of channels. '
                     'Use list_channels(cached=True) to avoid this warning.')
            if len(self.channels) > 0:
                return self.channels.keys()
            else:
                return []
        else:
            self.req_sock.send_pyobj(ChannelListRequest())
            return self.req_sock.recv_pyobj()

    def update_channels(self, channel_list, start=-1, end=-1, step=1):
        """
        Retrieves data for a specified set of channels and combines that data
        with any previously retrived data.

        This assumes all the channels have the same number of values. It is
        unclear as to whether this is a reasonable assumption. If they do not
        have the same number of values then it may request to much or too
        little data leading to duplicated data or wholes in the data
        respectively. This could be made more robust by making a call to
        retrieve all the data for all of the channels.

        Parameters
        ----------
        channel_list : list
            A list of the channels for which data should be requested.

        start : int
            The starting epoch for which data should be requested.

        step : int
            The number of epochs to be skipped between data points.
        """
        assert (start == -1 and end == -1) or end > start

        if self.subscribe:
            if start != -1 or end != -1 or step != 1:
                LOG.warn('Subscribe variant of LiveMonitor is only capable '
                         'of retreiving last result.'
                         'Use update_channels(channel_list) to avoid this warning.')
            # then again, we could see if we already have that range cached
            # locally and only throw the warning otherwise
        else:
            if start == -1:
                start = 0
                if len(self.channels.keys()) > 0:
                    channel_name = list(self.channels.keys())[0]
                    start = len(self.channels[channel_name].epoch_record)

            self.req_sock.send_pyobj(ChannelsRequest(
                channel_list, start=start, end=end, step=step
            ))

        rsp_msg = self.req_sock.recv_pyobj()

        if isinstance(rsp_msg.data, Exception):
            raise rsp_msg.data

        for channel in rsp_msg.data.keys():
            rsp_chan = rsp_msg.data[channel]

            if isinstance(rsp_chan, Exception):
                raise rsp_chan

            if self.subscribe:
                if channel not in channel_list:
                    continue

            if channel not in self.channels.keys():
                self.channels[channel] = rsp_chan
            else:
                chan = self.channels[channel]

                len_batch_rec = len(rsp_chan.batch_record)
                assert len_batch_rec == len(rsp_chan.epoch_record)
                assert len_batch_rec == len(rsp_chan.example_record)
                assert len_batch_rec == len(rsp_chan.time_record)
                assert len_batch_rec == len(rsp_chan.val_record)

                chan.batch_record += rsp_chan.batch_record
                chan.epoch_record += rsp_chan.epoch_record
                chan.example_record += rsp_chan.example_record
                chan.time_record += rsp_chan.time_record
                chan.val_record += rsp_chan.val_record

    def follow_channels(self, channel_list, use_qt=False):
        """
        Tracks and plots a specified set of channels in real time.

        Parameters
        ----------
        channel_list : list or dict
            A list of the channels for which data will be requested an plotted
            or a dictionary where keys will become the names of the plots while
            values are lists of channel names.
        use_qt : bool
            Use a PySide GUI for plotting, if available.
        """
        if use_qt:
            self.__qt_follow__(channel_list)

        elif not PYPLOT_AVAILABLE:
            raise ImportError('pyplot needs to be installed for '
                              'this functionality.')
        else:
            self.__ion_follow__(channel_list)

    def __ion_follow__(self, channel_list):
        """
        follow_channels() implementation using ion().
        """
        plt.clf()
        plt.ion()
        while True:
            self.update_channels(channel_list)
            plt.clf()
            for channel_name in self.channels:
                plt.plot(
                    self.channels[channel_name].epoch_record,
                    self.channels[channel_name].val_record,
                    label=channel_name
                )
            plt.legend()
            plt.ion()
            plt.draw()

    def __qt_follow__(self, channel_list):
        """
        follow_channels() implementation using Qt.
        """
        if not QT_AVAILABLE:
            LOG.warning(
                'follow_channels called with use_qt=True, but PySide '
                'is not available. Falling back on matplotlib ion().')
            self.__ion_follow__(channel_list)
        else:
            # only create new qt app if running the first time in session

            if isinstance(channel_list, dict):
                self.channel_dict = channel_list
                tmp_list = []
                for k in channel_list:
                    tmp_list.extend(channel_list[k])
                channel_list = tmp_list

                # remove duplicates in the list of channels
                self.channel_list = list(set(tmp_list))
            else:
                self.channel_list = channel_list
                self.channel_dict = {'': channel_list}

            if len(self.channel_list) == 0:
                raise ValueError('No channel name provided; '
                                 'channel_list must be either '
                                 'a list or a dict')

            if not hasattr(self, 'gui'):
                self.gui = LiveMonitorGUI(self,
                                          self.channel_list,
                                          self.channel_dict)

            self.gui.start()

if QT_AVAILABLE:

    class LiveMonitorGUI(QtGui.QMainWindow):
        """
        PySide GUI implementation for live monitoring channels.

        Parameters
        ----------
        live_mon : LiveMonitor instance
            The LiveMonitor instance to which the GUI belongs.

        channel_list : list
            A list of the channels to display.
        """
        def __init__(self, live_mon, channel_list, channel_dict):

            self.app = QtGui.QApplication(["Live Monitor"])

            super(LiveMonitorGUI, self).__init__()
            self.live_mon = live_mon
            self.channel_list = channel_list
            self.channel_dict = channel_dict
            self.updater_thread = UpdaterThread(live_mon, channel_list)
            self.updater_thread.updated.connect(self.__refresh__)
            self.__init_ui__()

        def __common_ui__(self):
            if MPLDC_AVAILABLE:
                opts = {'hover': True,
                        'xytext':(15, -30),
                        'formatter':"{label} {y:0.3g}\nat epoch {x:0.0f}".format,
                        'keybindings':{'hide':'h', 'toggle':'e'},
                        'bbox':{'fc':'white'},
                        'arrowprops': {'arrowstyle':'simple',
                                       'fc':'white',
                                       'alpha':0.1}}
                #draggable=True
                mpldc.datacursor(axes=self.fig.axes, **opts)

        def __init_ui__(self):
            matplotlib.rcParams.update({'font.size': 8})
            self.resize(600, 400)
            self.fig = Figure(figsize=(600, 400), dpi=72,
                              facecolor=(1, 1, 1), edgecolor=(0, 0, 0))

            arrange = {1: [1, 1], 2: [1, 2], 3: [1, 3], 4: [2, 2],
                       5: [2, 3], 6: [2, 3], 7: [2, 4], 8: [2, 4],
                       9: [3, 3], 10: [3, 4], 11: [3, 4], 12: [3, 4]}
            splot_len = len(self.channel_dict)
            if splot_len < 13:
                splot_layout = arrange[splot_len]
            else:
                splot_layout = [splot_len//5, 5]

            self.ax = []
            for splot_i in enumerate(len(self.channel_dict)):
                self.ax.append(self.fig.add_subplot(splot_layout[0],
                                                    splot_layout[1],
                                                    splot_i+1))

            self.fig.subplots_adjust(left=0.02, right=0.98,
                                     top=0.98, bottom=0.02,
                                     hspace=0.1)
            self.__common_ui__()
            self.canvas = FigureCanvas(self.fig)
            self.setCentralWidget(self.canvas)
            ntb = NavigationToolbar(self.canvas, self)
            self.addToolBar(ntb)

        def __refresh__(self):
            if not self.live_mon.channels:
                self.updater_thread.start()
                return

            splot_i = 0
            for splot_name in self.channel_dict:
                self.ax[splot_i].cla()  # clear previous plot
                chan_list = self.channel_dict[splot_name]

                for channel_name in chan_list:
                    if not channel_name in self.live_mon.channels:
                        splot_i = splot_i + 1
                        continue

                    X = epoch_record = self.live_mon.channels[channel_name].epoch_record
                    Y = val_record = self.live_mon.channels[channel_name].val_record

                    indices = np.nonzero(np.diff(epoch_record))[0] + 1
                    epoch_record_split = np.split(epoch_record, indices)
                    val_record_split = np.split(val_record, indices)

                    X = np.zeros(len(epoch_record))
                    Y = np.zeros(len(epoch_record))

                    for i, epoch in enumerate(epoch_record_split):

                        j = i*len(epoch_record_split[0])
                        X[j: j + len(epoch)] = (
                            1.*np.arange(len(epoch)) / len(epoch) + epoch[0])
                        Y[j: j + len(epoch)] = val_record_split[i]

                    self.ax[splot_i].plot(X, Y, label=channel_name)
                self.ax[splot_i].legend(loc='best', fancybox=True, framealpha=0.5)
                self.fig.axes[splot_i].set_xlabel('Epoch')
                self.fig.axes[splot_i].set_ylabel('Value')
                self.fig.axes[splot_i].set_title(splot_name)
                splot_i = splot_i + 1
                #self.fig.axes[splot_i].set_title('Tracking %d channels' % len(chan_list))

            self.__common_ui__()
            self.canvas.draw()
            self.updater_thread.start()

        def start(self):
            self.show()
            self.updater_thread.start()
            self.app.exec_()

    class UpdaterThread(QtCore.QThread):
        updated = QtCore.Signal()

        def __init__(self, live_mon, channel_list):
            super(UpdaterThread, self).__init__()
            self.live_mon = live_mon
            self.channel_list = channel_list

        def run(self):
            self.live_mon.update_channels(self.channel_list)  # blocking
            self.updated.emit()
