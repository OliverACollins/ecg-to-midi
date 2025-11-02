# ECG-to-MIDI
**The equivalent of EEG-to-MIDI, but for ECG.**

My attempt at creating an ECG-MIDI interface, whereby a participant's EEG signals would create/modulate music in Ableton Live.

My aim is to create functional bridge scripts for both (1) live ECG-to-MIDI conversion and (2) ECG-to-MIDI conversion for pre-recorded EEG data. The goal would be to test these interfaces within a biofeedback paradigm.


## Requirements
### Hardware
- PC/Laptop
- ECG device (if undertaking *live* ECG-to-MIDI) with biosignal acquisition kit (e.g., BITalino)

### Software
- VScode (with Python and Jupyter extensions)
- Python
- [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html)
- Ableton Live

## Proposed Setup: Live ECG-to-MIDI
1. Record live EEG signals through OpenSignals
2. Run a Python bridge script to extract the live ECG signals and convert them to MIDI output
3. Direct the MIDI output into loopMIDI, creating a virtual port
4. Send the information from loopMIDI into Ableton Live
5. Once the output is in Ableton Live, notes will be played/parameters will be modulated according to BPM change threshold (currently: +/- 2 BPM)

### Roadmap
- [ ] Create live bridge script that plays notes for each BPM change threshold
- [ ] Create live bridge script that modulates a quality (OPERATIONALISE!!!) of a note (e.g., default note could be C4, where an increase or decrease in heart rate leads to more or less gain/distortion of the tone)


## Proposed Setup: Pre-recorded ECG-to-MIDI
1. Locate .csv file containing (clean) ECG data
2. Run a Python bridge script to extract the pre-recorded ECG signals and convert them to MIDI output
3. Direct the MIDI output into loopMIDI, creating a virtual port
4. Send the information from loopMIDI into Ableton Live
5. Once the output is in Ableton Live, notes will be played/parameters will be modulated according to BPM change threshold (currently: +/- 2 BPM)

### Roadmap
- [x] Create pre-recorded bridge script that plays notes for each BPM change threshold, working for changes BOTH in increases and decreases of BPM
- [ ] Create pre-recorded bridge script that modulates a quality (OPERATIONALISE!!!) of a note (e.g., default note could be C4, where an increase or decrease in heart rate leads to more or less gain/distortion of the tone, OR it could be that a pre-recorded piece of music is played)

## Usage: Live ECG-to-MIDI

(TBC)

## Usage: Pre-recorded ECG-to-MIDI

(TBC)

Ideas for both live and pre-recorded ECG-to-MIDI conversion
- Change in BPM = change in pitch of note
- Change in BPM = more or less gain/distiortion
- Make a particularly relaxing version for biofeedback meditation session(?) - find an appropriate instrument on Ableton (e.g., ambient synth, marimba, acoustic instrument)
- Maybe make a paradigm focusing on HRV? Would need to be highly-sensitive to intervals between heart beats. Although, cannot really see any useful psychological applications of this idea

## Troubleshooting
- Create a new loopMIDI port each time the PC/laptop is restarted
- In Ableton, in the Preferences page, under the relevant loopMIDI input port, ensure that the "Track" and "Remote" boxes are ticked
