NTIA/SCOS Actions
====================

Base repository for creating new actions for [scos-sensor](https://github.com/NTIA/scos-sensor) and supporting new
hardware.

Requires pip>=18.1 (upgrade using `python3 -m pip install --upgrade pip`)

This repository includes the base [Action](scos_actions/actions/interfaces/action.py) class and offers 2 action classes for 
common measurements. The [acquire_single_freq_fft action](scos_actions/actions/acquire_single_freq_fft.py) is an action
class which performs FFTs and calculates mean, median, min, max, and sample statistics at a single center frequency.
The [acquire_stepped_freq_tdomain_iq action](scos_actions/actions/acquire_stepped_freq_tdomain_iq.py) acquires iq data
at 1 or more center frequencies.

3 parameterized actions using these action classes are
offered for testing using a mock radio. The 3 actions' parameters are defined in 
[configs/actions](scos_actions/configs/actions). The 3 actions are listed below:

* test_multi_frequency_iq_action
* test_single_frequency_iq_action
* test_single_frequency_m4s_action

This repository also contains an action class for getting gps location 
called sync_gps, and a monitor_radio action class for ensuring a radio
is available and is able to maintain a connection to the computer.

Adding Actions
--------------

To expose a new action to the API, check out the available [action
classes](scos_actions/actions). An _action_ is a parameterized implementation of
an action class. If an existing class covers your needs, you can simply add a text
[config file](scos_actions/configs/README.md) and restart the sensor.

If no existing action class meets your needs, see [Writing Custom
Actions](#writing-custom-actions).

Supporting a Different SDR
==========================

[scos_usrp](https://github.com/ntia/scos_usrp) adds support for the Ettus B2xx line of
software-defined radios to `scos-sensor`. If you want to use a different SDR that has a Python
API, you should able to do so with little effort:

 - Create a new repository called `scos_[SDR name]`
 - In the new repository, add this scos_actions repository as a dependency and create a class which inherits from the 
 [RadioInterface](scos_actions/hardware/radio_iface.py) abstract class.
 - Create `.yml` files with the parameters needed to run the actions using the new SDR. Put them in the new repository
   in `configs/actions`.
 - In the new repository, add a `discover/__init__.py` file. This should contain a dictionary called `actions` with a
   key of action name and a value of action object. You can use the [`init()`](scos_actions/discover/__init__.py)
   and/or the [`load_from_yaml()`](scos_actions/discover/yaml.py) methods provided in this repository to look for yaml
   files and initialize actions. These methods allow you to pass your new radio object to the action's constructor. You
   can use the existing action classes [defined in this repository](scos_actions/actions/__init__.py) or
   [create custom actions](#writing-custom-actions).

If your SDR doesn't have a Python API, you'll need a Python wrapper that
calls out to your SDR's available API and reads the samples back into Python. Libraries, such as
[SWIG](http://www.swig.org/), can automatically generate Python wrappers for programs written in C/C++.

The next step in supporting a different SDR would be to create a class which inherits from the
[GPSInterface](scos_actions/hardware/gps_iface.py) abstract class. Then add the sync_gps and monitor_radio actions to
your `actions` dictionary, passing the gps object to the SyncGps constructor, and the radio object to the RadioMonitor
constructor.

The final step would be add a setup.py to allow for installation of the new repository as a Python package. You can use
the [setup.py](setup.py) in this repository as a reference. You can find more information about Python packaging
[here](https://packaging.python.org/tutorials/packaging-projects/). Then add the new repository as a dependency to
[scos-sensor requirements.txt](https://github.com/NTIA/scos-sensor/blob/master/src/requirements.txt) using the
following format:  
`<package_name> @ git+<link_to_github_repo>@<branch_name>#egg=<package_name>`

If specific drivers are required for your SDR, you can attempt to link to them within the package or create a docker
image with the necessary files. You can host the docker image as a
[GitHub package](https://help.github.com/en/packages/using-github-packages-with-your-projects-ecosystem/configuring-docker-for-use-with-github-packages).
Then, when running scos-sensor, set the environment variable `BASE_IMAGE=<image tag>`.

Writing Custom Actions
======================

"Actions" are one of the main concepts used by [scos-sensor](https://github.com/NTIA/scos-sensor). At a high level,
they are the things that the sensor owner wants the sensor to be able to _do_. At a lower level, they are simply Python
classes with a special method `__call__`. Actions use
[Django Signals](https://docs.djangoproject.com/en/3.0/topics/signals/) to provide data and results to scos-sensor.

Start by looking at the [Action base class](scos_actions/actions/interfaces/action.py). It includes some logic to parse
a description and summary out of the action class's docstring, and a `__call__` method that must be overridden. If you
pass `admin_only=True` to this base class, the API will not make it or any data it created available to non-admin users.

Depending on the type of action, a signal should be sent upon action completion. This enables scos-sensor to do
something with the results of the action. This could range from storing measurement data to recycling a 
docker container to fix an unhealthy connection to the SDR. You can see the available signals
in [scos_actions/actions/interfaces/signals.py](scos_actions/actions/interfaces/signals.py). The following signals are
currently offered:
 * `measurement_action_completed` - signal expects task_id, data, and metadata
 * `location_action_completed` - signal expects latitude and longitude
 * `monitor_action_completed` - signal expects boolean called radio_healthy
 
New signals can be added. However, corresponding signal handlers must be added to scos-sensor to receive the signals
and process the results.

Lastly, to expose a custom action to the API and make it schedulable,
instantiate it in the `actions` dict in the discover module,
[here](scos_actions/discover/__init__.py). To use yaml initialization, create the desired yaml files in
[scos_actions/configs/actions](scos_actions/configs/actions) and add the action class to `action_classes` in
[scos_actions/actions/\_\_init\_\_.py](scos_actions/actions/__init__.py). The custom action can instead be put into a
new repository as long as it is follows the actions discovery convention where the action name and object are added to
a dictionary in `package_name/discover/__init__.py` and the package name starts with `scos_`.