# -*- coding: utf-8 -*-
'''
Experiment of Rhythm Temporal Attention 
Optimized Version
'''
from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
import pandas as pd
import numpy as np
import time, os, random
import copy
import subprocess

# =============================================================================
# 1. Path and Environment Configuration 
# (Optimization: Use relative paths and explicit VLC path)
# =============================================================================
# Get current script directory for portability
root_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_dir)

# [IMPORTANT] Modify vlc_path according to your actual system configuration
# Default installation path is usually C:\Program Files\VideoLAN\VLC\vlc.exe
vlc_path = r'C:\Program Files\VideoLAN\VLC\vlc.exe'
if not os.path.exists(vlc_path):
    # Try another common path
    vlc_path = r'D:\VLC\vlc.exe'
    if not os.path.exists(vlc_path):
        print(f"Warning: VLC not found at {vlc_path}, video playback may fail.")

# %% Record Information
expInfo = {'observer':'01_TEST', 
            'gender':'',
            'age':'', 
            'fraRate':100,
            'emo':'ne', # Options: po, ne, neu
            'contrast':0.5,
            }
dateStr = time.strftime("_20%y_%m_%d_%H%M", time.localtime())

# Fullscreen mode (0=windowed, 1=fullscreen)
isfullscr = 0

# %% Set experiment parameters
dlg = gui.DlgFromDict(expInfo, 
                      title='DNMoA with emotion', 
                      order=['observer',
                             'gender',
                             'age', 
                             'fraRate',
                             'emo',
                             'contrast'])
if dlg.OK: 
    print(expInfo)
else: 
    core.quit()


# Process Emotion labels
if expInfo['emo'] == 'po':
    emoType = 'Positive'
elif expInfo['emo'] == 'ne':
    emoType = 'Negative'
elif expInfo['emo'] == 'neu':
    emoType = 'Neutral'
print(f"Current Emotion Condition: {emoType}")

# Create the subject datafile directory
print(expInfo['gender'])
print(str(expInfo['age']))

new_path = os.path.join(root_dir, 'data', 
                        str(expInfo['observer'])+'_'+str(expInfo['gender'])+'_'+str(expInfo['age']))
if not os.path.isdir(new_path): 
    os.makedirs(new_path)

dataFN = os.path.join(new_path, str(emoType)+str(expInfo['contrast'])+dateStr+'_')
fn1 = dataFN + '.csv'
fn_conditions = dataFN + 'conditions.csv'

# =============================================================================
# 2. Condition Generation Function
# =============================================================================
def make_conditions(emo_type, nReps=1, vid_path=None, pic_path=None, file_name=None):
    conditions = []
    for targetDir in ['left', 'right']:
        for validity in ['valid', 'invalid']:
            for thissoa in [300, 900]:
                if validity == 'valid':
                    soa = 600
                elif validity == 'invalid':
                    soa = thissoa
                
                orientations = np.arange(1.6, 2.4, 0.1)
                ori = round(orientations[np.random.randint(0, len(orientations))], 1)
                if targetDir == 'left':
                    ori = -ori
                randFix = int(round(random.randint(500, 1300), -1))
                
                temp = {'targetDir': targetDir, 
                        'validity': validity, 
                        'soa': soa,
                        'ori': ori,
                        'randFix': randFix,
                        'emotype': emo_type
                        }
                
                # Adjust trial proportions
                if validity == 'valid':
                    for i in range(3):
                        conditions.append(temp)
                elif validity == 'invalid':
                    for i in range(2):
                        conditions.append(temp)

    conditions = conditions * nReps
    
    # Emotion video path assignment
    if vid_path == None:
        if emo_type == 'Positive':
            vid_path = ['positive_1', 'positive_2']
        elif emo_type == 'Negative':
            vid_path = ['negative_1', 'negative_2']
        elif emo_type == 'Neutral':
            vid_path = ['neutral_1', 'neutral_2']
    
    random.shuffle(vid_path)
    
    final_lists = []
    for vid in vid_path:
        random.shuffle(conditions)
        for condition in conditions:
            temp = copy.deepcopy(condition)
            temp['vid_path'] = vid
            final_lists.append(temp)
   
    condition_file = pd.DataFrame(final_lists)
    if file_name and file_name.endswith('.csv'):
        condition_file.to_csv(file_name, index=False, encoding='utf-8')
    elif file_name and file_name.endswith('.xlsx'):
        condition_file.to_excel(file_name, index=False, engine='openpyxl')
    return final_lists

# Generate conditions
conditions = make_conditions(emoType, file_name=fn_conditions, nReps=5) 
nRep = 1
trial_list = data.TrialHandler(nReps=nRep, method='random', trialList=conditions)
print(f"Total Trials: {trial_list.nTotal}")

# %% Set stimulus presentation time and conditions
frameRate = expInfo['fraRate']
frameDura = 1000/frameRate 

def time2frames(time, frameDura=10):
    frames = int(round(time/frameDura, 0))
    return frames

class durations:
    def __init__(self):
        self.fixDura = 1000      
        self.targetDura = 30     
        self.stimDura = 100      
        self.ITIDura = 600      
        self.FeedbackDura = 1000   
        self.emoDura = 2500

dr = durations()
framesFix = time2frames(dr.fixDura, frameDura)
framesTarget = time2frames(dr.targetDura, frameDura)
framesStim = time2frames(dr.stimDura, frameDura)
framesITI = time2frames(dr.ITIDura, frameDura)
framesFeedback = time2frames(dr.FeedbackDura, frameDura)
framesEmo = time2frames(dr.emoDura, frameDura)

# =============================================================================
# 3. Initialize Window and Stimuli 
# (Optimization: Global initialization to avoid recreation in loops)
# =============================================================================
scnWidth, scnHeight = (1024, 768)
center = (0.0, 0.0)
mon = monitors.Monitor('win', width=64.0, distance=56)
mon.setSizePix((scnWidth, scnHeight))

win = visual.Window((scnWidth, scnHeight), 
                    screen=0,
                    color=(128, 128, 128), 
                    fullscr=isfullscr, 
                    units='pix', blendMode='avg', colorSpace='rgb255', monitor=mon)

# Pre-define all visual stimulus objects
fix = visual.TextStim(win, text='+', pos=center, color='black', bold=True, height=30)  

introduction_path = os.path.join(root_dir, 'materials', 'intr.png')
if os.path.exists(introduction_path):
    intr = visual.ImageStim(win, image=introduction_path, size=(scnWidth, scnHeight), pos=center, units='pix')
else:
    # Fallback if image missing
    intr = visual.TextStim(win, text='Instructions (Image Missing)', color='black')

# Prompt for rest period
restPrompt = visual.TextStim(win, text=u'请闭眼休息两分钟！', color='black', bold=True, pos=(0, 0), height=50)
# Prompt for block start
blockPrompt = visual.TextStim(win, text=u'按空格键开始', color='black', bold=True, pos=(0, -60), height=50)
# Prompt for experiment end
overPrompt = visual.TextStim(win, text=u'实验结束，谢谢！', color='black', bold=True, pos=(0, 0), height=50)

correctFB = visual.TextStim(win, text=u'CORRECT', color='green', bold=True, pos=(0, 0), height=20)
wrongFB = visual.TextStim(win, text=u'WRONG', color='red', bold=True, pos=(0, 0), height=20)

stimSize = (70, 70)
whiteRect = visual.Rect(win, size=stimSize, units='pix', lineColor='white', fillColor='white', pos=center)
redRect = visual.Rect(win, size=stimSize, units='pix', lineColor='red', fillColor='red', pos=center)

# Optimization: Initialize target here instead of inside run_trial
target = visual.GratingStim(win=win, mask='gauss', tex='sin', 
                            size=(70, 70), sf=0.04, pos=(0, 0))

# =============================================================================
# 4. Trial Execution Function
# =============================================================================
def run_trial(contrast, randFix, validity, soa, targetDir, ori, nRhythms=3):
    # Update Target attributes
    target.ori = ori
    target.contrast = contrast
    
    framesRandFix = time2frames(randFix, frameDura)
    
    # Present fixation
    for frameN in range(framesRandFix):
        fix.draw()
        win.flip()

    # Present Rhythm Cues
    for thisRhythm in range(nRhythms):
        for frameN in range(framesStim):
            redRect.draw()
            win.flip()
        for frameN in range(framesITI):
            win.flip() # Blank screen interval
            
    # Alert signal
    for frameN in range(framesStim):
        whiteRect.draw()
        win.flip()
        
    # SOA (Inter-Stimulus Interval)
    ISI = time2frames(soa, frameDura)
    for frameN in range(ISI):
        win.flip()
    
    # Present Target
    for frameN in range(framesStim):
        target.draw()
        win.flip()

    # Response phase
    event.clearEvents()        
    response = None
    frameN = 0
    frameRT = 0
    
    while response is None:
        fix.draw()
        frameN += 1
        win.flip()
        
        keys = event.getKeys()
        if 'left' in keys:
            response = 'left'
            frameRT = frameN
            break
        elif 'right' in keys:
            response = 'right'
            frameRT = frameN
            break
        elif frameN > time2frames(5000, frameDura=frameDura):
            response = 'No Response'
            frameRT = frameN
            break
            
    rt = np.round(frameRT * frameDura)
    
    # Determine correctness
    res = 0
    if (response == 'left' and targetDir == 'left') or \
       (response == 'right' and targetDir == 'right'):
        res = 1
    
    # Feedback
    for frameN in range(framesFeedback):
        if res == 1:
            correctFB.draw()
        else:
            wrongFB.draw()
        win.flip()
        
    fix.draw() # Draw fixation to clear feedback
    return [res, rt, contrast, randFix, validity, soa, targetDir, ori, nRhythms]

# =============================================================================
# 5. Session Execution Function
# =============================================================================
def run_session(contrast):
    isRunning = 1
    nBlocks = 1
    nTrials = np.ceil(trial_list.nTotal/nBlocks)
    response_data = [] # Rename variable to prevent confusion
    
    # Temporary variable for image loading
    emoPic = []

    for trial in trial_list:
        # Check for stop signal
        if isRunning == 0:
            break
            
        randFix = trial['randFix']
        validity = trial['validity']
        invalidFix = trial['soa'] # Note: Key is 'soa', corresponding to logic in make_conditions
        targetDir = trial['targetDir']
        ori = trial['ori']
        
        # --- Experiment Start/Block Logic ---
        if trial_list.thisN == 0: 
            event.clearEvents()
            intr.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            
            # Play video
            emopath = trial['vid_path']
            emo_vid_path = os.path.join(root_dir, 'materials', 'emo', emopath+'.mp4')
            
            print(f"Attempting to play: {emo_vid_path}")
            try:
                # Optimization: Add cwd and wait
                subprocess.Popen([vlc_path, '--video-on-top', '--fullscreen', '--play-and-exit', emo_vid_path], 
                                 cwd=os.path.dirname(emo_vid_path))
                # Critical: Allow time for VLC to start, preventing premature clearEvents
                core.wait(0.5) 
                event.clearEvents()
            except Exception as e:
                print(f"Video playback error: {e}")

            # Prepare image paths
            emo_pic_path1 = os.path.join(root_dir, 'materials', 'emo', emopath+'_1.png')
            emo_pic_path2 = os.path.join(root_dir, 'materials', 'emo', emopath+'_2.png')
            emoPic = [emo_pic_path1, emo_pic_path2]
            
            blockPrompt.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            
        # --- Rest Logic (Every 30 trials) ---
        elif (trial_list.thisN % 30 == 0) and not (trial_list.thisN + 1 == trial_list.nTotal): 
            # Randomly display an emotion image
            if random.random() > 0.5:
                curr_pic_path = emoPic[0]
            else:
                curr_pic_path = emoPic[1]
            
            # Check if image exists
            if os.path.exists(curr_pic_path):
                emo_pic_stim = visual.ImageStim(win, image=curr_pic_path, size=(scnWidth, scnHeight), pos=center, units='pix')
                # Display for 3x fixation duration
                for frameN in range(framesFix * 3):
                    emo_pic_stim.draw()
                    win.flip()
            
            # Rest prompt
            restPrompt.draw()
            blockPrompt.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            
            # Replay video after rest (maintain original logic)
            emopath = trial['vid_path']
            emo_vid_path = os.path.join(root_dir, 'materials', 'emo', emopath+'.mp4')
            try:
                subprocess.Popen([vlc_path, '--video-on-top', '--fullscreen', '--play-and-exit', emo_vid_path],
                                 cwd=os.path.dirname(emo_vid_path))
                core.wait(0.5)
                event.clearEvents()
            except Exception as e:
                 print(e)
                 
            win.flip()
            # Update image paths in case of changes (safety check)
            emo_pic_path1 = os.path.join(root_dir, 'materials', 'emo', emopath+'_1.png')
            emo_pic_path2 = os.path.join(root_dir, 'materials', 'emo', emopath+'_2.png')
            emoPic = [emo_pic_path1, emo_pic_path2]
            event.waitKeys(keyList=['space'])

        # --- Run Trial ---
        ans = run_trial(contrast, randFix, validity, invalidFix, targetDir, ori)
        ans = ans + [emoType] # Add emotion label
        response_data.append(ans) # Manually add to List
        
        # --- Keyboard Detection (Escape) ---
        keys = event.getKeys()
        if 'escape' in keys:
            print("Escape pressed. Saving data...")
            # Save data immediately
            df_res = pd.DataFrame(response_data)
            df_res.columns = ['res', 'rt', 'contrast', 'FixDura', 'validity', 'soa', 'targetDir', 'ori', 'nRhythms', 'emo']
            df_res.to_csv(fn1, index=False)
            isRunning = 0
            win.close()
            core.quit()
        
        # --- End Logic ---
        if (trial_list.thisN + 1) == trial_list.nTotal:
            overPrompt.draw()
            win.flip()
            core.wait(2) # Shorten wait time slightly
            isRunning = 0

    return response_data

# =============================================================================
# 6. Main Program Entry
# =============================================================================
if __name__ == '__main__':
    answer_list = run_session(expInfo['contrast'])
    
    # Final data saving
    if answer_list:
        answer_df = pd.DataFrame(answer_list)
        answer_df.columns = ['res', 'rt', 'contrast', 'FixDura', 'validity', 'soa', 'targetDir', 'ori', 'nRhythms', 'emo']
        answer_df.to_csv(fn1, index=False)
        print(f"Data saved to {fn1}")
    
    win.close()
    core.quit()
