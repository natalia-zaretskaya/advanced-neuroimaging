#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on Tue Apr 28 13:21:54 2026
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
expName = 'mct'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '001',
    'session': '01',
    'run': '01',
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
        originPath='/Users/natalia/Nextcloud/current_teaching/2026_SS_SE_ANI/experiments/mct/mct_lastrun.py',
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
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
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
    # Make folder to store recordings from idea_mic
    idea_micRecFolder = filename + '_idea_mic_recorded'
    if not os.path.isdir(idea_micRecFolder):
        os.mkdir(idea_micRecFolder)
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
    
    # --- Initialize components for Routine "settings" ---
    # Run 'Begin Experiment' code from code
    ##  enable audio recording  ##
    #from psychopy import microphone
    #microphone.switchOn()
    #idea_mic = microphone.AdvAudioCapture(stereo=False, chnl=0)
    
    #print("filename"); print(filename)
    #wavDirName = filename + '_wav'
    #if not os.path.isdir(wavDirName):
    #    os.makedirs(wavDirName)  # to hold .wav files
    #print("wavDirName"); print(wavDirName)
    
    
    ##  Options for DEBUG/RESARCH-mode  ##
    DEBUG = 0   #0=research-mode; 1=debug-mode 
    if DEBUG:
        #loopDur     = 10 #total time for ideas
        #p_loopDur   = 10 #total time for practice-ideas
        p_audioDur  = 5 #duration for p-audio
        itemDur     = 5
        fixDur      = 1
        #recDur      = 3 #orig 10s sound/answer recording
    else:
        #loopDur     = 180 #total time for ideas
        #p_loopDur   = 60 #total time for ideas
        p_audioDur  = 34 #duration for p-audio
        itemDur     = 115
        fixDur      = 30
        #recDur      = 10 #orig 10s sound/answer recording
    
    
    # --- Initialize components for Routine "wait_for_mri" ---
    wait_text = visual.TextStim(win=win, name='wait_text',
        text='Bitte warten.\n\nDas Experiment startet in Kürze!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "t_items" ---
    t_fix = visual.TextStim(win=win, name='t_fix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t_img = visual.ImageStim(
        win=win,
        name='t_img', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # set audio backend
    sound.Sound.backend = 'ptb'
    t_audio = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='t_audio'
    )
    t_audio.setVolume(1.0)
    
    # --- Initialize components for Routine "t_idea" ---
    t_FZ = visual.TextStim(win=win, name='t_FZ',
        text='?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text = visual.TextStim(win=win, name='text',
        text='?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='green', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # make microphone object for idea_mic
    idea_mic = sound.microphone.Microphone(
        device=None,
        name='idea_mic',
        recordingFolder=idea_micRecFolder,
        recordingExt='wav'
    )
    # tell the experiment handler to save this Microphone's clips if the experiment is force ended
    runAtExit.append(idea_mic.saveClips)
    # connect camera save method to experiment handler so it's called when data saves
    thisExp.connectSaveMethod(idea_mic.saveClips)
    
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
    
    # --- Prepare to start Routine "settings" ---
    # create an object to store info about Routine settings
    settings = data.Routine(
        name='settings',
        components=[],
    )
    settings.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for settings
    settings.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    settings.tStart = globalClock.getTime(format='float')
    settings.status = STARTED
    thisExp.addData('settings.started', settings.tStart)
    settings.maxDuration = None
    # keep track of which components have finished
    settingsComponents = settings.components
    for thisComponent in settings.components:
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
    
    # --- Run Routine "settings" ---
    thisExp.currentRoutine = settings
    settings.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
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
                currentRoutine=settings,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            settings.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if settings.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in settings.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "settings" ---
    for thisComponent in settings.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for settings
    settings.tStop = globalClock.getTime(format='float')
    settings.tStopRefresh = tThisFlipGlobal
    thisExp.addData('settings.stopped', settings.tStop)
    thisExp.nextEntry()
    # the Routine "settings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "wait_for_mri" ---
    # create an object to store info about Routine wait_for_mri
    wait_for_mri = data.Routine(
        name='wait_for_mri',
        components=[wait_text, key_resp],
    )
    wait_for_mri.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for wait_for_mri
    wait_for_mri.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    wait_for_mri.tStart = globalClock.getTime(format='float')
    wait_for_mri.status = STARTED
    thisExp.addData('wait_for_mri.started', wait_for_mri.tStart)
    wait_for_mri.maxDuration = None
    # keep track of which components have finished
    wait_for_mriComponents = wait_for_mri.components
    for thisComponent in wait_for_mri.components:
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
    
    # --- Run Routine "wait_for_mri" ---
    thisExp.currentRoutine = wait_for_mri
    wait_for_mri.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *wait_text* updates
        
        # if wait_text is starting this frame...
        if wait_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            wait_text.frameNStart = frameN  # exact frame index
            wait_text.tStart = t  # local t and not account for scr refresh
            wait_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wait_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wait_text.started')
            # update status
            wait_text.status = STARTED
            wait_text.setAutoDraw(True)
        
        # if wait_text is active this frame...
        if wait_text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['5'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
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
                currentRoutine=wait_for_mri,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            wait_for_mri.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if wait_for_mri.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in wait_for_mri.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "wait_for_mri" ---
    for thisComponent in wait_for_mri.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for wait_for_mri
    wait_for_mri.tStop = globalClock.getTime(format='float')
    wait_for_mri.tStopRefresh = tThisFlipGlobal
    thisExp.addData('wait_for_mri.stopped', wait_for_mri.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "wait_for_mri" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(f"stim/sub-{int(expInfo['participant']):03d}/ses-{int(expInfo['session']):02d}/sub-{int(expInfo['participant']):03d}_ses-{int(expInfo['session']):02d}_MCT_run-{int(expInfo['run']):02d}_conditions.csv"), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "t_items" ---
        # create an object to store info about Routine t_items
        t_items = data.Routine(
            name='t_items',
            components=[t_fix, t_img, t_audio],
        )
        t_items.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        t_img.setImage(item)
        t_audio.setSound(audiofile, secs=audio_dur, hamming=True)
        t_audio.setVolume(1.0, log=False)
        t_audio.seek(0)
        # store start times for t_items
        t_items.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        t_items.tStart = globalClock.getTime(format='float')
        t_items.status = STARTED
        thisExp.addData('t_items.started', t_items.tStart)
        t_items.maxDuration = None
        # keep track of which components have finished
        t_itemsComponents = t_items.components
        for thisComponent in t_items.components:
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
        
        # --- Run Routine "t_items" ---
        thisExp.currentRoutine = t_items
        t_items.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *t_fix* updates
            
            # if t_fix is starting this frame...
            if t_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_fix.frameNStart = frameN  # exact frame index
                t_fix.tStart = t  # local t and not account for scr refresh
                t_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_fix.started')
                # update status
                t_fix.status = STARTED
                t_fix.setAutoDraw(True)
            
            # if t_fix is active this frame...
            if t_fix.status == STARTED:
                # update params
                pass
            
            # if t_fix is stopping this frame...
            if t_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_fix.tStartRefresh + fixDur-frameTolerance:
                    # keep track of stop time/frame for later
                    t_fix.tStop = t  # not accounting for scr refresh
                    t_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    t_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_fix.stopped')
                    # update status
                    t_fix.status = FINISHED
                    t_fix.setAutoDraw(False)
            
            # *t_img* updates
            
            # if t_img is starting this frame...
            if t_img.status == NOT_STARTED and t_fix.status==FINISHED:
                # keep track of start time/frame for later
                t_img.frameNStart = frameN  # exact frame index
                t_img.tStart = t  # local t and not account for scr refresh
                t_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_img.started')
                # update status
                t_img.status = STARTED
                t_img.setAutoDraw(True)
            
            # if t_img is active this frame...
            if t_img.status == STARTED:
                # update params
                pass
            
            # if t_img is stopping this frame...
            if t_img.status == STARTED:
                if bool(t_audio.status==FINISHED):
                    # keep track of stop time/frame for later
                    t_img.tStop = t  # not accounting for scr refresh
                    t_img.tStopRefresh = tThisFlipGlobal  # on global time
                    t_img.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_img.stopped')
                    # update status
                    t_img.status = FINISHED
                    t_img.setAutoDraw(False)
            
            # *t_audio* updates
            
            # if t_audio is starting this frame...
            if t_audio.status == NOT_STARTED and tThisFlip >= fixDur-frameTolerance:
                # keep track of start time/frame for later
                t_audio.frameNStart = frameN  # exact frame index
                t_audio.tStart = t  # local t and not account for scr refresh
                t_audio.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('t_audio.started', tThisFlipGlobal)
                # update status
                t_audio.status = STARTED
                t_audio.play(when=win)  # sync with win flip
            
            # if t_audio is stopping this frame...
            if t_audio.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_audio.tStartRefresh + audio_dur-frameTolerance or t_audio.isFinished:
                    # keep track of stop time/frame for later
                    t_audio.tStop = t  # not accounting for scr refresh
                    t_audio.tStopRefresh = tThisFlipGlobal  # on global time
                    t_audio.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_audio.stopped')
                    # update status
                    t_audio.status = FINISHED
                    t_audio.stop()
            
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
                    currentRoutine=t_items,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                t_items.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if t_items.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in t_items.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "t_items" ---
        for thisComponent in t_items.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for t_items
        t_items.tStop = globalClock.getTime(format='float')
        t_items.tStopRefresh = tThisFlipGlobal
        thisExp.addData('t_items.stopped', t_items.tStop)
        t_audio.pause()  # ensure sound has stopped at end of Routine
        try:
            if t_audio.tStopRefresh is not None:
                duration_val = t_audio.tStopRefresh - t_audio.tStartRefresh
            else:
                duration_val = thisExp.thisEntry['t_items.stopped'] - t_audio.tStartRefresh
            bids_event = BIDSTaskEvent(
                onset=t_audio.tStartRefresh-key_resp.tStartRefresh-key_resp.rt,
                duration=duration_val,
                event_type='audio',
                trial_type=condition,
            )
            if bids_handler:
                bids_handler.addEvent(bids_event)
            else:
                trials.addData('bidsEvent_audio.event', bids_event)
        except BIDSError as e:
            print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
        # the Routine "t_items" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        item_loop = data.TrialHandler2(
            name='item_loop',
            nReps=3, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(item_loop)  # add the loop to the experiment
        thisItem_loop = item_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisItem_loop.rgb)
        if thisItem_loop != None:
            for paramName in thisItem_loop:
                globals()[paramName] = thisItem_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisItem_loop in item_loop:
            item_loop.status = STARTED
            if hasattr(thisItem_loop, 'status'):
                thisItem_loop.status = STARTED
            currentLoop = item_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisItem_loop.rgb)
            if thisItem_loop != None:
                for paramName in thisItem_loop:
                    globals()[paramName] = thisItem_loop[paramName]
            
            # --- Prepare to start Routine "t_idea" ---
            # create an object to store info about Routine t_idea
            t_idea = data.Routine(
                name='t_idea',
                components=[t_FZ, text, idea_mic],
            )
            t_idea.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            idea_mic.setPolicyWhenFull('warn')
            # store start times for t_idea
            t_idea.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            t_idea.tStart = globalClock.getTime(format='float')
            t_idea.status = STARTED
            thisExp.addData('t_idea.started', t_idea.tStart)
            t_idea.maxDuration = None
            # keep track of which components have finished
            t_ideaComponents = t_idea.components
            for thisComponent in t_idea.components:
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
            
            # --- Run Routine "t_idea" ---
            thisExp.currentRoutine = t_idea
            t_idea.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 26.0:
                # if trial has changed, end Routine now
                if hasattr(thisItem_loop, 'status') and thisItem_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *t_FZ* updates
                
                # if t_FZ is starting this frame...
                if t_FZ.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    t_FZ.frameNStart = frameN  # exact frame index
                    t_FZ.tStart = t  # local t and not account for scr refresh
                    t_FZ.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_FZ, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_FZ.started')
                    # update status
                    t_FZ.status = STARTED
                    t_FZ.setAutoDraw(True)
                
                # if t_FZ is active this frame...
                if t_FZ.status == STARTED:
                    # update params
                    pass
                
                # if t_FZ is stopping this frame...
                if t_FZ.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > t_FZ.tStartRefresh + 15-frameTolerance:
                        # keep track of stop time/frame for later
                        t_FZ.tStop = t  # not accounting for scr refresh
                        t_FZ.tStopRefresh = tThisFlipGlobal  # on global time
                        t_FZ.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 't_FZ.stopped')
                        # update status
                        t_FZ.status = FINISHED
                        t_FZ.setAutoDraw(False)
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # if text is stopping this frame...
                if text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text.tStartRefresh + 11-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.tStopRefresh = tThisFlipGlobal  # on global time
                        text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.stopped')
                        # update status
                        text.status = FINISHED
                        text.setAutoDraw(False)
                
                # if idea_mic is starting this frame...
                if idea_mic.status == NOT_STARTED and t >= 15-frameTolerance:
                    # keep track of start time/frame for later
                    idea_mic.frameNStart = frameN  # exact frame index
                    idea_mic.tStart = t  # local t and not account for scr refresh
                    idea_mic.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(idea_mic, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('idea_mic.started', t)
                    # update status
                    idea_mic.status = STARTED
                    # start recording with idea_mic
                    idea_mic.start()
                
                # if idea_mic is active this frame...
                if idea_mic.status == STARTED:
                    # update params
                    pass
                    # update recorded clip for idea_mic
                    idea_mic.poll()
                
                # if idea_mic is stopping this frame...
                if idea_mic.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > idea_mic.tStartRefresh + 11-frameTolerance:
                        # keep track of stop time/frame for later
                        idea_mic.tStop = t  # not accounting for scr refresh
                        idea_mic.tStopRefresh = tThisFlipGlobal  # on global time
                        idea_mic.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.addData('idea_mic.stopped', t)
                        # update status
                        idea_mic.status = FINISHED
                        # stop recording with idea_mic
                        idea_mic.stop()
                
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
                        currentRoutine=t_idea,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    t_idea.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if t_idea.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in t_idea.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "t_idea" ---
            for thisComponent in t_idea.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for t_idea
            t_idea.tStop = globalClock.getTime(format='float')
            t_idea.tStopRefresh = tThisFlipGlobal
            thisExp.addData('t_idea.stopped', t_idea.tStop)
            # tell mic to keep hold of current recording in idea_mic.clips and transcript (if applicable) in idea_mic.scripts
            # this will also update idea_mic.lastClip and idea_mic.lastScript
            idea_mic.stop()
            tag = data.utils.getDateStr()
            idea_micClip = idea_mic.bank(
                tag=tag, transcribe='None',
                config=None
            )
            item_loop.addData(
                'idea_mic.clip', idea_mic.recordingFolder / idea_mic.getClipFilename(tag)
            )
            try:
                if t_FZ.tStopRefresh is not None:
                    duration_val = t_FZ.tStopRefresh - t_FZ.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['t_idea.stopped'] - t_FZ.tStartRefresh
                bids_event = BIDSTaskEvent(
                    onset=t_FZ.tStartRefresh-key_resp.tStartRefresh-key_resp.rt,
                    duration=duration_val,
                    event_type='thinking',
                    trial_type=condition,
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    item_loop.addData('bidsEvent_tFZ.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            try:
                if idea_mic.tStopRefresh is not None:
                    duration_val = idea_mic.tStopRefresh - idea_mic.tStartRefresh
                else:
                    duration_val = thisExp.thisEntry['t_idea.stopped'] - idea_mic.tStartRefresh
                bids_event = BIDSTaskEvent(
                    onset=idea_mic.tStartRefresh-key_resp.tStartRefresh-key_resp.rt,
                    duration=duration_val,
                    event_type='speaking',
                    trial_type=condition,
                )
                if bids_handler:
                    bids_handler.addEvent(bids_event)
                else:
                    item_loop.addData('bidsEvent_idea_mic.event', bids_event)
            except BIDSError as e:
                print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if t_idea.maxDurationReached:
                routineTimer.addTime(-t_idea.maxDuration)
            elif t_idea.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-26.000000)
            # mark thisItem_loop as finished
            if hasattr(thisItem_loop, 'status'):
                thisItem_loop.status = FINISHED
            # if awaiting a pause, pause now
            if item_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                item_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 3 repeats of 'item_loop'
        item_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    thisExp.nextEntry()
    # the Routine "bidsExport" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # save idea_mic recordings
    idea_mic.saveClips()
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
            bids_handler.writeEvents(participant_info, add_stimuli=True, execute_sidecar=True, generate_hed_metadata=False)
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
