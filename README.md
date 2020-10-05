# NTIA/SCOS Actions

Base repository for creating new actions for
[scos-sensor](https://github.com/NTIA/scos-sensor) and supporting new hardware.

Requires pip>=18.1 (upgrade using `python3 -m pip install --upgrade pip`) and
python>=3.6

This repository includes the base [Action](scos_actions/actions/interfaces/action.py)
class and offers 3 action classes for common measurements. The [acquire_single_freq_fft
action](scos_actions/actions/acquire_single_freq_fft.py) is an action class which
performs FFTs and calculates mean, median, min, max, and sample statistics at a single
center frequency. The [acquire_single_freq_tdomain_iq action](
    scos_actions/actions/acquire_single_freq_tdomain_iq.py) acquires iq data at a
single center frequency. The [acquire_stepped_freq_tdomain_iq action](
    scos_actions/actions/acquire_stepped_freq_tdomain_iq.py) acquires iq data at
multiple center frequencies.

3 parameterized actions using these action classes are offered for testing using a mock
radio. The 3 actions' parameters are defined in [configs/actions](
    scos_actions/configs/actions). The 3 actions are listed below:

- test_multi_frequency_iq_action
- test_single_frequency_iq_action
- test_single_frequency_m4s_action

This repository also contains an action class for getting GPS location and syncing the
host to GPS time called sync_gps, and a monitor_radio action class for ensuring a radio
is available and is able to maintain a connection to the computer.

## Running in scos-sensor

1. Clone scos-sensor: `git clone https://github.com/NTIA/scos-sensor.git`
1. Navigate to scos-sensor: `cd scos-sensor`
1. Checkout SMBWTB475_refactor_radio_interface branch:
    `git checkout SMBWTB475_refactor_radio_interface`
1. In scos-sensor/src/requirements.txt, comment out the following line:
    `scos_usrp @ git+${DOCKER_GIT_CREDENTIALS}/NTIA/scos_usrp@master#egg=scos_usrp`
1. If you are using a different branch than master, change master in the following line
    to the branch you are using:
    `scos_actions @
    git+${DOCKER_GIT_CREDENTIALS}/NTIA/scos-actions@master#egg=scos_actions`
1. `cp env.template ./env`
1. In env file, change `BASE_IMAGE=ubuntu:18.04` (at the bottom of the file)
1. If you are already using git credential manager (you have the file
    ~/.git-credentials), you can skip this step. In the env file, change
   `DOCKER_GIT_CREDENTIALS=https://<username>:<password>@github.com`
1. Set MOCK_RADIO and MOCK_RADIO_RANDOM equal to 1 in docker-compose.yml
1. Get environment variables: `source ./env`
1. Start services: `docker-compose up -d --build --force-recreate`
