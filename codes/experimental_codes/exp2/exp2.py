# -*- coding: utf-8 -*-
'''
Experiment of Rhythm Temporal Attention 
Optimized version
'''
from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import pandas as pd
import numpy as np
import time, os, random
import subprocess

# ========================== Path Check Optimization ==========================
# Check VLC path to prevent immediate crashes if not found
default_vlc = r'C:\Program Files\VideoLAN\VLC\vlc.exe'
backup_vlc = r'D:\VLC\vlc.exe'

if os.path.exists(default_vlc):
    vlc = default_vlc
elif os.path.exists(backup_vlc):
    vlc = backup_vlc
else:
    print("Warning: VLC player not found! Video playback might fail.")
    vlc = 'vlc' # Attempt to call from system environment variables

dateStr = time.strftime("_20%y_%m_%d_%H%M", time.localtime())
total_trials = 200

# ========================== Parameter Settings ==========================
pickleName = 'exp3.pickle'
try:
    expInfo = fromFile(pickleName)       
except: 
    expInfo = {
        'subject': '00_test',
        'gender': 'm',
        'age': "",
        'nRhythms': 3,
        'emo':['n','p','neu', 'NULL'], # n=negative, p=positive, neu=neutral
        'screenRate': 100,
        'contrast': 0.6
    }

dlg = gui.DlgFromDict(expInfo, title='exp:3', order=['subject','gender','age', 'nRhythms','emo','screenRate','contrast'])
if dlg.OK: 
    toFile(pickleName, expInfo)
else: 
    core.quit()

# ========================== Stimuli Generation (Opt: Remove 'ori' param) ==========================
def create_stimuli(win):
    # Only create objects here; specific parameters are modified at runtime
    return {
        'fixation': visual.TextStim(win, text='+', height=30, color='white'),
        'rhythm': visual.Rect(win, size=(100,100), fillColor='red'),
        'alert': visual.Rect(win, size=(100,100), fillColor='white'),
        'target': visual.GratingStim(win, mask='gauss', tex='sin', 
                    size=65, sf=0.04, contrast=expInfo['contrast'],
                    ori=0), # Initial ori set to 0, modified later
        'blockPrompt': visual.TextStim(win, text='Press Space to Start', color='black',
                                       bold=True, pos=(0,-60), height=50),
        'overPrompt': visual.TextStim(win, text='Experiment Ended. Thank you!', color='black', 
                                      bold=True, pos=(0,0), height=50),
        # Optimization: Pre-create an ImageStim object for emotional pictures
        'emo_placeholder': visual.ImageStim(win, pos=(0, 0), units='pix') 
    }

# File Settings
dataDir = os.path.join(os.getcwd(), 'data')
sub_dataDir = os.path.join(dataDir, f"{expInfo['subject']}_{expInfo['age']}_{expInfo['gender']}")
if not os.path.exists(sub_dataDir): os.makedirs(sub_dataDir)

# Time Parameters
class TimeParams:
    def __init__(self, frameRate):
        self.frameDura = 1000 / frameRate
        self.fixation = 1000
        self.rhythm = 100
        self.interval = 600
        self.alert = 100
        self.target = 30
        self.feedback = 500
        self.ITI = 1000

    def to_frames(self, duration):
        return int(round(duration / self.frameDura))

# ========================== Trial Workflow ==========================
def run_trial(win, stimuli, params, nRhythms, validity, direction, ori):
    # Optimization: Update stimulus parameters here instead of recreating them
    stimuli['target'].ori = ori
    
    response = {'RTtime': np.nan, 'RTframes': np.nan, 'correct': 0}
    
    # Fixation
    for thisFrame in range(params.to_frames(params.fixation)):
        stimuli['fixation'].draw()
        win.flip()

    # Rhythm Cues
    if validity == "neutral":
        for this_n in range(nRhythms):
            for thisFrame in range(params.to_frames(params.rhythm)):
                stimuli['rhythm'].draw()
                win.flip()
            interval_time = params.interval + random.choice(np.arange(-300,300,50))
            for thisFrame in range(params.to_frames(interval_time)):
                win.flip()
    else:
        for this_n in range(nRhythms):
            for thisFrame in range(params.to_frames(params.rhythm)):
                stimuli['rhythm'].draw()
                win.flip()
            for thisFrame in range(params.to_frames(params.interval)):
                win.flip()

    # Alert Signal
    for thisFrame in range(params.to_frames(params.alert)):
        stimuli['alert'].draw()
        win.flip()

    # Target Validity Time Control
    targetTime = (params.interval)
    if validity == 'invalid':
        targetTime += random.choice([-300, 300])

    for _ in range(params.to_frames(targetTime)):
        win.flip()

    # Show Target
    event.clearEvents()
    for _ in range(params.to_frames(params.target)):
        stimuli['target'].draw()
        win.flip()
    
    # Response Phase
    event.clearEvents() # Ensure keys pressed during target display are cleared
    frameCount = 0
    startTime = core.getTime()
    responded = False
    rt = np.nan
    correct = 0
    
    # Optimization: Avoid repeated access to params.to_frames inside the loop
    max_frames = params.to_frames(5000)
    
    while not responded:
        frameCount += 1
        stimuli['fixation'].draw()
        win.flip()
        
        keys = event.getKeys(keyList=['left','right'], timeStamped=True)
        if keys:
            rt = keys[0][1] - startTime
            if keys[0][0] == direction:
                correct = 1 
            else:
                correct = 0
            responded = True
        if frameCount > max_frames:
            rt = 5
            correct = 2 # 2 indicates timeout/miss
            break
        
    if keys:
        response['RTtime'] = np.round(rt*1000)
        response['RTframes'] = np.round(frameCount * params.frameDura)
        response['correct'] = correct
        response['targettime'] = targetTime

    # Feedback
    fb = visual.TextStim(win, text='correct' if response['correct'] == 1 else 'wrong', 
                         color='green' if response['correct'] == 1 else 'red')
    for _ in range(params.to_frames(params.feedback)):
        fb.draw()
        win.flip()

    for _ in range(params.to_frames(params.ITI)):
        win.flip()

    return response


# ========================== Condition Generation ==========================
def generate_conditions(total_trials=200):
    # Logic remains unchanged
    valid_prop = 0.6
    invalid_prop = 0.3
    neutral_prop = 1- (invalid_prop + valid_prop)
    validity_conds = [
        ('valid', valid_prop, [600]),
        ('invalid', invalid_prop, [300, 900]),
        ('neutral', neutral_prop, range(300, 901, 100))
    ]
    directions = ['left', 'right']
    oris = np.arange(1.2, 2, 0.1)
    
    counts = {
        'valid': int(total_trials * valid_prop),
        'invalid': int(total_trials * invalid_prop),
        'neutral': total_trials - int(total_trials*(invalid_prop + valid_prop))
    }
    
    conditions = []
    for cond_type, ratio, times in validity_conds:
        for i in range(counts[cond_type]):
            direction = directions[i % 2]
            ori = random.choice(oris) * (-1 if direction == 'left' else 1)
            if cond_type == 'valid':
                target_time = 600
            else:
                target_time = random.choice(times)
            
            conditions.append({
                'validity': cond_type,
                'targetDir': direction,
                'targetOri': round(ori, 1),
                'targetTime': target_time,
                'nRhythms': int(expInfo['nRhythms'])
            })
    return conditions

# ========================== Main Program ==========================
if __name__ == "__main__":
    win = visual.Window([1024,768], fullscr=1, color='gray', units='pix')
    win.mouseVisible = False # Hide mouse
    frameRate = win.getActualFrameRate()
    if frameRate is None: frameRate = 60 # Fallback default
    params = TimeParams(frameRate)
    
    conditions = generate_conditions(total_trials=total_trials)
    trials = data.TrialHandler(trialList=conditions, nReps=1, method='random')
    
    # Optimization: Initialize all stimulus objects outside the loop
    all_stimuli = create_stimuli(win)
    
    # Instructions
    intro_path = os.path.join('materials', 'intro.jpg')
    if os.path.exists(intro_path):
        intro = visual.ImageStim(win, image=intro_path, size=(1024, 768), pos=(0, 0), units='pix')
        intro.draw()
    else:
        all_stimuli['blockPrompt'].text = 'Press Space to Continue' # Fallback text
        all_stimuli['blockPrompt'].draw()
        
    win.flip()
    event.clearEvents()
    keys = event.waitKeys(keyList=['space', 'escape'])

    # Emotion material logic (Unchanged)
    if expInfo['emo'] == 'n':
        emo = 'negative'
        emoFiles = ['negative_1', 'negative_2', 'negative_3', 'negative_4', 'negative_1']
    elif expInfo['emo'] == 'p':
        emo = 'positive'
        emoFiles = ['positive_1', 'positive_2', 'positive_1', 'positive_2', 'positive_1']
    elif expInfo['emo'] == 'neu':
        emo = 'neutral'
        emoFiles = ['neutral_1', 'neutral_2', 'neutral_1', 'neutral_2', 'neutral_1']
    else:
        emo = 'null'
        emoFiles = ['neutral_1'] * 5 # Fallback

    # Index out of bounds protection
    rhythm_idx = max(0, min(int(expInfo['nRhythms'])-1, len(emoFiles)-1))
    emoFile = emoFiles[rhythm_idx]

    emopath = os.path.join(os.getcwd(), 'materials','emo', emoFile+'.mp4')
    emoPic1 = os.path.join(os.getcwd(), 'materials','emo', emoFile+'_1.png')
    emoPic2 = os.path.join(os.getcwd(), 'materials','emo', emoFile+'_1.png') # Note: Original code duplicated _1.png here
    emoPics = [emoPic1, emoPic2]
    
    event.clearEvents()
    event.waitKeys(keyList=['space'])
    
    if os.path.exists(emopath):
        try:
            # Play video. Suggest replacing subprocess with psychopy.visual.MovieStim3 in the future
            subprocess.Popen([vlc, '--video-on-top', '--fullscreen','--play-and-exit',emopath])
            core.wait(1) # Give VLC time to launch
        except Exception as e:
            print(f"Video error: {e}")
    
    event.clearEvents()
    event.waitKeys(keyList=['space'])
    
    emoPicN = random.choice([28,29,31,32])
    
    dat = []
    
    # Start Loop
    for trial in trials:
        # Optimization: Pass the pre-created all_stimuli and the current trial's ori
        response = run_trial(win, all_stimuli, params, 
                             trial['nRhythms'], 
                             trial['validity'],
                             trial['targetDir'],
                             trial['targetOri']) # New parameter
                             
        # Record Data
        this_data = {
            **response, 
            'nRhythms': trial['nRhythms'], 
            'validity': trial['validity'],
            'direction': trial['targetDir'],
            'ori': trial['targetOri'],
            'emo': emo
        }
        dat.append(this_data)
        
        # Optimization: Use TrialHandler's built-in save function (double insurance)
        for k, v in this_data.items():
            trials.addData(k, v)
            
        # Rest Break
        if (trials.thisN + 1) % 60 == 0 and (trials.thisN + 1) != trials.nTotal:
            rest_text = visual.TextStim(win, text="Please rest. Press Space to continue.", color='white', height=40, pos=(0,0))
            continue_text = visual.TextStim(win, text="Trials remaining: " + str(trials.nTotal-trials.thisN-1), color='white', height=30, pos=(0,-100))      
            event.clearEvents()
            while True:
                rest_text.draw()
                continue_text.draw()
                win.flip() 
                if 'space' in event.getKeys(keyList=['space', 'escape']):
                    break       
        
        # Insert Emotional Picture
        if (trials.thisN + 1) % emoPicN == 0 and (trials.thisN + 1) != trials.nTotal :
            # Optimization: Reuse emo_placeholder, only update image attribute
            pic_path = random.choice(emoPics)
            if os.path.exists(pic_path):
                all_stimuli['emo_placeholder'].image = pic_path
                all_stimuli['emo_placeholder'].size = (1024, 768)
                
                for frame in range(params.to_frames(2500)):
                    all_stimuli['emo_placeholder'].draw()
                    win.flip()
            else:
                print(f"Image not found: {pic_path}")
    
        # Exit Detection
        if 'escape' in event.getKeys():
            print("Escape pressed. Saving data...")
            break
    
    # Experiment End
    for frame in range(params.to_frames(2000)):
        all_stimuli['overPrompt'].draw()
        win.flip()
        
    # Save Data
    # Method 1: Original Pandas save
    dat_df = pd.DataFrame(dat)
    csv_name = os.path.join(sub_dataDir, f"{expInfo['subject']}_{emo}_{expInfo['nRhythms']}_{dateStr}.csv")
    dat_df.to_csv(csv_name, index=False)
    
    # Method 2: PsychoPy built-in save (Recommended, includes more metadata)
    # trials.saveAsWideText(csv_name.replace('.csv', '_psy.csv'))
    
    win.close()
    core.quit()
