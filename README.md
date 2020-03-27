NTIA/SCOS Actions
====================

Base repository for creating new actions for scos-sensor and supporting new hardware.

Requires pip>=18.1 (upgrade using `python3 -m pip install --upgrade pip`)

This repository includes the base Action class and offers 2 action class for 
common measurements. 3 parameterized actions using these action classes are
offered for testing using a mock RadioInterface. The 3 actions' parameters are defined in 
[configs/actions](scos_actions/configs/actions). The 3 actions are listed below:

* test_multi_frequency_iq_action
* test_single_frequency_iq_action
* test_single_frequency_m4s_action

This repository also contains a action class for getting gps location 
called sync_gps, and a monitor_radio action class for ensuring a radio
is available and maintains a good connection to the computer.

Adding Actions
--------------

To expose a new action to the API, check out the available [action
classes](scos_actions/actions). An _action_ is a parameterized implementation of
an action class. If an existing class covers your needs, you can simply add a text
[config file](scos_actions/actions/README.md) and restart the sensor.

If no existing action class meets your needs, see [Writing Custom
Actions](#writing-custom-actions).

Supporting a Different SDR
==========================

`scos-usrp` adds support for the Ettus B2xx line of
software-defined radios to `scos-sensor`. If you want to use a different SDR that has a python
API, you should able to do so with little effort:

 - Create a new repository called `scos_[SDR name]`
 - In the new repository, implement the [RadioInterface](scos_actions/hardware/radio_iface.py)
 - Create `.yml` files with the parameters needed to run the actions using the new SDR. Put them in the new repository
   in `configs/actions`.
 - In the new repository, add a `discover/__init__.py` file. This should contain a dictionary with a key of action name
   and a value of action object. You can use the `init()` and/or the `load_from_yaml()` methods in `scos_actions` in
   `discover` to read the yaml files and initialize the action objects. These methods allow you to pass your new
   RadioInterface implementation object.

If your SDR doesn't have a python API, you'll need a python adapater file that
calls out to your SDRs available API and reads the samples back into python.

The next step in supporting a different SDR would be to modify the
[monitor_radio](scos_actions/actions/monitor_radio.py) action which can be used to
periodically exercise the SDR and signal Docker to recycle the container
if its connection is lost. You may also want to modify the [sync_gps](scos_actions/actions/sync_gps.py) action if your
SDR supports GPS. Next we'll go into more depth about _actions_ and how to write them.


Writing Custom Actions
======================

"Actions" are one of the main concepts used by `scos-sensor`. At a high level,
they are the things that the sensor owner wants the sensor to be able to _do_.
At a lower level, they are simply python classes with a special method
`__call__`. Actions use [Django Signals](https://docs.djangoproject.com/en/3.0/topics/signals/) to provide data and
results to scos-sensor.

Start by looking at the [Action base class](scos_actions/actions/interfaces/action.py). It includes
some logic to parse a description and summary out of the action class's
docstring, and a `__call__` method that must be overridden. If you pass
`admin_only=True` to this base class, the API will not make it or any data it
created available to non-admin users.

Depending on the type of action, a signal should be sent upon action completion. This enables scos-sensor to do
something with the results of the action. This could range from storing measurement data to recycling a 
docker container to fix an unhealthy connection to the SDR. You can see the available signals
in [scos_actions/actions/interfaces/signals.py](scos_actions/actions/interfaces/signals.py). The following signals are
currently offered:
 * `measurement_action_completed` - signal expects task_id, data, and metadata
 * `location_action_completed` - signal expects latitude and longitude
 * `monitor_action_completed` - signal expects radio_healthy
 
 New signals can be added. However, corresponding signal handlers must be added to scos-sensor to receive the signals and
 process the results.
 

The [acquire_single_freq_fft
action](scos_actions/actions/acquire_single_freq_fft.py) is an action classes which calculates mean, median, min, max, and
sample statistics at a single center frequency.

The [acquire_stepped_freq_tdomain_iq action](scos_actions/actions/acquire_stepped_freq_tdomain_iq.py) aquires iq data
at 1 or more center frequencies.

Lastly, to expose a custom action to the API and make it schedulable,
instantiate it in the `actions` dict in the discover module,
[here](scos_actions/discover/__init__.py). To use yaml initialization, create the desired yaml files in
[scos_actions/configs/actions](scos_actions/configs/actions) and add the action class to `action_classes` in
[scos_actions/actions/__init__.py](scos_actions/actions/__init__.py)