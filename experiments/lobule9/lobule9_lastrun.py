#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on Tue Apr 28 14:40:32 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard
from psychopy_bids.bids import BIDSBehEvent
from psychopy_bids.bids import BIDSTaskEvent
from psychopy_bids.bids import BIDSError
from psychopy_bids.bids import BIDSHandler

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'lobule9'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '001',
    'session': '01',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/natalia/Nextcloud/current_teaching/2026_SS_SE_ANI/experiments/lobule9/lobule9_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1,1,1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1,1,1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    if expInfo['session']:
        bids_handler = BIDSHandler(dataset='bids',
         subject=expInfo['participant'], task=expInfo['expName'],
         session=expInfo['session'], data_type='func', acq='',
         runs=True)
    else:
        bids_handler = BIDSHandler(dataset='bids',
         subject=expInfo['participant'], task=expInfo['expName'],
         data_type='func', acq='', runs=True)
    bids_handler.createDataset()
    bids_handler.addTaskCode(force=True)
    
    # --- Initialize components for Routine "indicator_start" ---
    start_indicator = visual.TextStim(win=win, name='start_indicator',
        text='Please wait, the experiment is starting…',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "audio" ---
    # set audio backend
    sound.Sound.backend = 'ptb'
    audio_trial = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='audio_trial'
    )
    audio_trial.setVolume(1.0)
    text_4 = visual.TextStim(win=win, name='text_4',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "question" ---
    question_trial = visual.TextStim(win=win, name='question_trial',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    resp_trial = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "break_3" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "GoodbyeScreen" ---
    Goodbye_text = visual.TextStim(win=win, name='Goodbye_text',
        text='This run is over, thank you!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "indicator_start" ---
    # create an object to store info about Routine indicator_start
    indicator_start = data.Routine(
        name='indicator_start',
        components=[start_indicator, start_resp],
    )
    indicator_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_resp
    start_resp.keys = []
    start_resp.rt = []
    _start_resp_allKeys = []
    # store start times for indicator_start
    indicator_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    indicator_start.tStart = globalClock.getTime(format='float')
    indicator_start.status = STARTED
    thisExp.addData('indicator_start.started', indicator_start.tStart)
    indicator_start.maxDuration = None
    # keep track of which components have finished
    indicator_startComponents = indicator_start.components
    for thisComponent in indicator_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "indicator_start" ---
    thisExp.currentRoutine = indicator_start
    indicator_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_indicator* updates
        
        # if start_indicator is starting this frame...
        if start_indicator.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_indicator.frameNStart = frameN  # exact frame index
            start_indicator.tStart = t  # local t and not account for scr refresh
            start_indicator.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_indicator, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_indicator.started')
            # update status
            start_indicator.status = STARTED
            start_indicator.setAutoDraw(True)
        
        # if start_indicator is active this frame...
        if start_indicator.status == STARTED:
            # update params
            pass
        
        # *start_resp* updates
        waitOnFlip = False
        
        # if start_resp is starting this frame...
        if start_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_resp.frameNStart = frameN  # exact frame index
            start_resp.tStart = t  # local t and not account for scr refresh
            start_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_resp.started')
            # update status
            start_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_resp.status == STARTED and not waitOnFlip:
            theseKeys = start_resp.getKeys(keyList=['5'], ignoreKeys=["escape"], waitRelease=False)
            _start_resp_allKeys.extend(theseKeys)
            if len(_start_resp_allKeys):
                start_resp.keys = _start_resp_allKeys[-1].name  # just the last key pressed
                start_resp.rt = _start_resp_allKeys[-1].rt
                start_resp.duration = _start_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=indicator_start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            indicator_start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if indicator_start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in indicator_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "indicator_start" ---
    for thisComponent in indicator_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for indicator_start
    indicator_start.tStop = globalClock.getTime(format='float')
    indicator_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('indicator_start.stopped', indicator_start.tStop)
    # check responses
    if start_resp.keys in ['', [], None]:  # No response was made
        start_resp.keys = None
    thisExp.addData('start_resp.keys',start_resp.keys)
    if start_resp.keys != None:  # we had a response
        thisExp.addData('start_resp.rt', start_resp.rt)
        thisExp.addData('start_resp.duration', start_resp.duration)
    thisExp.nextEntry()
    # the Routine "indicator_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_list = data.TrialHandler2(
        name='trial_list',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trial_list)  # add the loop to the experiment
    thisTrial_list = trial_list.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_list.rgb)
    if thisTrial_list != None:
        for paramName in thisTrial_list:
            globals()[paramName] = thisTrial_list[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_list in trial_list:
        trial_list.status = STARTED
        if hasattr(thisTrial_list, 'status'):
            thisTrial_list.status = STARTED
        currentLoop = trial_list
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_list.rgb)
        if thisTrial_list != None:
            for paramName in thisTrial_list:
                globals()[paramName] = thisTrial_list[paramName]
        
        # --- Prepare to start Routine "audio" ---
        # create an object to store info about Routine audio
        audio = data.Routine(
            name='audio',
            components=[audio_trial, text_4],
        )
        audio.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        audio_trial.setSound(file, secs=7.0, hamming=True)
        audio_trial.setVolume(1.0, log=False)
        audio_trial.seek(0)
        # store start times for audio
        audio.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        audio.tStart = globalClock.getTime(format='float')
        audio.status = STARTED
        thisExp.addData('audio.started', audio.tStart)
        audio.maxDuration = None
        # keep track of which components have finished
        audioComponents = audio.components
        for thisComponent in audio.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "audio" ---
        thisExp.currentRoutine = audio
        audio.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_list, 'status') and thisTrial_list.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *audio_trial* updates
            
            # if audio_trial is starting this frame...
            if audio_trial.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                audio_trial.frameNStart = frameN  # exact frame index
                audio_trial.tStart = t  # local t and not account for scr refresh
                audio_trial.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('audio_trial.started', tThisFlipGlobal)
                # update status
                audio_trial.status = STARTED
                audio_trial.play(when=win)  # sync with win flip
            
            # if audio_trial is stopping this frame...
            if audio_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > audio_trial.tStartRefresh + 7.0-frameTolerance or audio_trial.isFinished:
                    # keep track of stop time/frame for later
                    audio_trial.tStop = t  # not accounting for scr refresh
                    audio_trial.tStopRefresh = tThisFlipGlobal  # on global time
                    audio_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'audio_trial.stopped')
                    # update status
                    audio_trial.status = FINISHED
                    audio_trial.stop()
            
            # *text_4* updates
            
            # if text_4 is starting this frame...
            if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.started')
                # update status
                text_4.status = STARTED
                text_4.setAutoDraw(True)
            
            # if text_4 is active this frame...
            if text_4.status == STARTED:
                # update params
                pass
            
            # if text_4 is stopping this frame...
            if text_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_4.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    text_4.tStop = t  # not accounting for scr refresh
                    text_4.tStopRefresh = tThisFlipGlobal  # on global time
                    text_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_4.stopped')
                    # update status
                    text_4.status = FINISHED
                    text_4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=audio,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                audio.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if audio.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in audio.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "audio" ---
        for thisComponent in audio.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for audio
        audio.tStop = globalClock.getTime(format='float')
        audio.tStopRefresh = tThisFlipGlobal
        thisExp.addData('audio.stopped', audio.tStop)
        audio_trial.pause()  # ensure sound has stopped at end of Routine
        try:
            if audio_trial.tStopRefresh is not None:
                duration_val = audio_trial.tStopRefresh - audio_trial.tStartRefresh
            else:
                duration_val = thisExp.thisEntry['audio.stopped'] - audio_trial.tStartRefresh
            bids_event = BIDSTaskEvent(
                onset=audio_trial.tStartRefresh-start_resp.tStartRefresh-start_resp.rt,
                duration=duration_val,
                trial_type=condition_type,
            )
            if bids_handler:
                bids_handler.addEvent(bids_event)
            else:
                trial_list.addData('bidsEvent_audio.event', bids_event)
        except BIDSError as e:
            print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if audio.maxDurationReached:
            routineTimer.addTime(-audio.maxDuration)
        elif audio.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "question" ---
        # create an object to store info about Routine question
        question = data.Routine(
            name='question',
            components=[question_trial, resp_trial],
        )
        question.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        question_trial.setText(question_text)
        # create starting attributes for resp_trial
        resp_trial.keys = []
        resp_trial.rt = []
        _resp_trial_allKeys = []
        # store start times for question
        question.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        question.tStart = globalClock.getTime(format='float')
        question.status = STARTED
        thisExp.addData('question.started', question.tStart)
        question.maxDuration = None
        # keep track of which components have finished
        questionComponents = question.components
        for thisComponent in question.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question" ---
        thisExp.currentRoutine = question
        question.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_list, 'status') and thisTrial_list.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_trial* updates
            
            # if question_trial is starting this frame...
            if question_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_trial.frameNStart = frameN  # exact frame index
                question_trial.tStart = t  # local t and not account for scr refresh
                question_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_trial.started')
                # update status
                question_trial.status = STARTED
                question_trial.setAutoDraw(True)
            
            # if question_trial is active this frame...
            if question_trial.status == STARTED:
                # update params
                pass
            
            # if question_trial is stopping this frame...
            if question_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > question_trial.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    question_trial.tStop = t  # not accounting for scr refresh
                    question_trial.tStopRefresh = tThisFlipGlobal  # on global time
                    question_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question_trial.stopped')
                    # update status
                    question_trial.status = FINISHED
                    question_trial.setAutoDraw(False)
            
            # *resp_trial* updates
            waitOnFlip = False
            
            # if resp_trial is starting this frame...
            if resp_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                resp_trial.frameNStart = frameN  # exact frame index
                resp_trial.tStart = t  # local t and not account for scr refresh
                resp_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(resp_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'resp_trial.started')
                # update status
                resp_trial.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(resp_trial.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(resp_trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if resp_trial is stopping this frame...
            if resp_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > resp_trial.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    resp_trial.tStop = t  # not accounting for scr refresh
                    resp_trial.tStopRefresh = tThisFlipGlobal  # on global time
                    resp_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp_trial.stopped')
                    # update status
                    resp_trial.status = FINISHED
                    resp_trial.status = FINISHED
            if resp_trial.status == STARTED and not waitOnFlip:
                theseKeys = resp_trial.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _resp_trial_allKeys.extend(theseKeys)
                if len(_resp_trial_allKeys):
                    resp_trial.keys = _resp_trial_allKeys[-1].name  # just the last key pressed
                    resp_trial.rt = _resp_trial_allKeys[-1].rt
                    resp_trial.duration = _resp_trial_allKeys[-1].duration
                    # was this correct?
                    if (resp_trial.keys == str('response')) or (resp_trial.keys == 'response'):
                        resp_trial.corr = 1
                    else:
                        resp_trial.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=question,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                question.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if question.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in question.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question" ---
        for thisComponent in question.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for question
        question.tStop = globalClock.getTime(format='float')
        question.tStopRefresh = tThisFlipGlobal
        thisExp.addData('question.stopped', question.tStop)
        # check responses
        if resp_trial.keys in ['', [], None]:  # No response was made
            resp_trial.keys = None
            # was no response the correct answer?!
            if str('response').lower() == 'none':
               resp_trial.corr = 1;  # correct non-response
            else:
               resp_trial.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_list (TrialHandler)
        trial_list.addData('resp_trial.keys',resp_trial.keys)
        trial_list.addData('resp_trial.corr', resp_trial.corr)
        if resp_trial.keys != None:  # we had a response
            trial_list.addData('resp_trial.rt', resp_trial.rt)
            trial_list.addData('resp_trial.duration', resp_trial.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if question.maxDurationReached:
            routineTimer.addTime(-question.maxDuration)
        elif question.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "break_3" ---
        # create an object to store info about Routine break_3
        break_3 = data.Routine(
            name='break_3',
            components=[text_3],
        )
        break_3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for break_3
        break_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        break_3.tStart = globalClock.getTime(format='float')
        break_3.status = STARTED
        thisExp.addData('break_3.started', break_3.tStart)
        break_3.maxDuration = None
        # keep track of which components have finished
        break_3Components = break_3.components
        for thisComponent in break_3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_3" ---
        thisExp.currentRoutine = break_3
        break_3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_list, 'status') and thisTrial_list.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # if text_3 is stopping this frame...
            if text_3.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > iti-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.stopped')
                    # update status
                    text_3.status = FINISHED
                    text_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=break_3,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                break_3.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if break_3.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in break_3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_3" ---
        for thisComponent in break_3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for break_3
        break_3.tStop = globalClock.getTime(format='float')
        break_3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('break_3.stopped', break_3.tStop)
        try:
            bids_event = BIDSTaskEvent(
                onset=text_3.tStartRefresh-start_resp.tStartRefresh-start_resp.rt,
                duration=0.1,
                trial_type='keypress',
            )
            if bids_handler:
                bids_handler.addEvent(bids_event)
            else:
                trial_list.addData('bidsEvent_buttonPress.event', bids_event)
        except BIDSError as e:
            print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
        # the Routine "break_3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial_list as finished
        if hasattr(thisTrial_list, 'status'):
            thisTrial_list.status = FINISHED
        # if awaiting a pause, pause now
        if trial_list.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trial_list.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trial_list'
    trial_list.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "GoodbyeScreen" ---
    # create an object to store info about Routine GoodbyeScreen
    GoodbyeScreen = data.Routine(
        name='GoodbyeScreen',
        components=[Goodbye_text],
    )
    GoodbyeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for GoodbyeScreen
    GoodbyeScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    GoodbyeScreen.tStart = globalClock.getTime(format='float')
    GoodbyeScreen.status = STARTED
    thisExp.addData('GoodbyeScreen.started', GoodbyeScreen.tStart)
    GoodbyeScreen.maxDuration = None
    # keep track of which components have finished
    GoodbyeScreenComponents = GoodbyeScreen.components
    for thisComponent in GoodbyeScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "GoodbyeScreen" ---
    thisExp.currentRoutine = GoodbyeScreen
    GoodbyeScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Goodbye_text* updates
        
        # if Goodbye_text is starting this frame...
        if Goodbye_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Goodbye_text.frameNStart = frameN  # exact frame index
            Goodbye_text.tStart = t  # local t and not account for scr refresh
            Goodbye_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Goodbye_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Goodbye_text.started')
            # update status
            Goodbye_text.status = STARTED
            Goodbye_text.setAutoDraw(True)
        
        # if Goodbye_text is active this frame...
        if Goodbye_text.status == STARTED:
            # update params
            pass
        
        # if Goodbye_text is stopping this frame...
        if Goodbye_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Goodbye_text.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                Goodbye_text.tStop = t  # not accounting for scr refresh
                Goodbye_text.tStopRefresh = tThisFlipGlobal  # on global time
                Goodbye_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Goodbye_text.stopped')
                # update status
                Goodbye_text.status = FINISHED
                Goodbye_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=GoodbyeScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            GoodbyeScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if GoodbyeScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in GoodbyeScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "GoodbyeScreen" ---
    for thisComponent in GoodbyeScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for GoodbyeScreen
    GoodbyeScreen.tStop = globalClock.getTime(format='float')
    GoodbyeScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('GoodbyeScreen.stopped', GoodbyeScreen.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if GoodbyeScreen.maxDurationReached:
        routineTimer.addTime(-GoodbyeScreen.maxDuration)
    elif GoodbyeScreen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    thisExp.nextEntry()
    # the Routine "bidsExport" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    ignore_list = [
        'participant',
        'session',
        'date',
        'expName',
        'psychopyVersion',
        'OS',
        'frameRate'
    ]
    participant_info = {
        key: thisExp.extraInfo[key]
        for key in thisExp.extraInfo
        if key not in ignore_list
    }
    # write tsv file and update
    try:
        if bids_handler.events:
            bids_handler.writeEvents(participant_info, add_stimuli=True, execute_sidecar=True, generate_hed_metadata=True)
    except Exception as e:
        print(f"[psychopy-bids(settings)] An error occurred when writing BIDS events: {e}")
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
