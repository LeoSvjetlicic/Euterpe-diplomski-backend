const Midi = require('jsmidgen');

const TOKEN_MAP = {
  0: 'blank',
  1: 'note_C4',
  2: 'note_D4',
  3: 'note_E4',
  4: 'note_F4',
  5: 'note_G4',
  6: 'note_A4',
  7: 'note_B4',
  8: 'note_C5',
  9: 'rest_quarter',
  10: 'rest_eighth',
  11: 'barline',
  12: 'time_4_4',
  13: 'key_C_major'
};

const NOTE_PITCHES = {
  'note_C4': 60,
  'note_D4': 62,
  'note_E4': 64,
  'note_F4': 65,
  'note_G4': 67,
  'note_A4': 69,
  'note_B4': 71,
  'note_C5': 72
};

async function convertToMidi(modelOutput) {
  try {
    if (!modelOutput.success) {
      throw new Error(`Model prediction failed: ${modelOutput.error}`);
    }
    
    const tokens = modelOutput.tokens;
    const musicElements = parseTokens(tokens);
    const midiBuffer = createMidiFile(musicElements);
    
    return midiBuffer;
    
  } catch (error) {
    throw new Error(`MIDI conversion failed: ${error.message}`);
  }
}

function parseTokens(tokens) {
  const musicElements = [];
  let currentTime = 0;
  let currentTimeSignature = '4/4';
  let currentKey = 'C';
  
  for (const token of tokens) {
    const tokenName = TOKEN_MAP[token] || 'unknown';
    
    if (tokenName.startsWith('note_')) {
      const pitch = NOTE_PITCHES[tokenName];
      if (pitch) {
        musicElements.push({
          type: 'note',
          pitch: pitch,
          startTime: currentTime,
          endTime: currentTime + 0.5, // Default quarter note duration
          velocity: 80
        });
        currentTime += 0.5;
      }
    } else if (tokenName.startsWith('rest_')) {
      const duration = tokenName === 'rest_quarter' ? 0.5 : 0.25;
      currentTime += duration;
    } else if (tokenName === 'barline') {
      // Bar lines don't affect timing in this simple implementation
      musicElements.push({
        type: 'barline',
        time: currentTime
      });
    } else if (tokenName.startsWith('time_')) {
      currentTimeSignature = tokenName.replace('time_', '').replace('_', '/');
      musicElements.push({
        type: 'timeSignature',
        time: currentTime,
        signature: currentTimeSignature
      });
    } else if (tokenName.startsWith('key_')) {
      const keyInfo = tokenName.replace('key_', '').split('_');
      currentKey = keyInfo[0];
      musicElements.push({
        type: 'keySignature',
        time: currentTime,
        key: currentKey,
        mode: keyInfo[1] || 'major'
      });
    }
  }
  
  return musicElements;
}

function createMidiFile(musicElements) {
  const file = new Midi.File();
  const track = new Midi.Track();
  file.addTrack(track);
  
  // Set tempo (120 BPM)
  track.setTempo(120);
  
  // Add time signature (4/4 by default)
  track.addEvent(new Midi.Event({
    type: Midi.Event.TIME_SIGNATURE,
    data: [4, 2, 24, 8] // 4/4 time
  }));
  
  const notes = musicElements.filter(el => el.type === 'note');
  
  // Sort notes by start time
  notes.sort((a, b) => a.startTime - b.startTime);
  
  for (const note of notes) {
    const startTicks = Math.round(note.startTime * 480); // 480 ticks per quarter note
    const duration = Math.round((note.endTime - note.startTime) * 480);
    
    track.addNote(0, note.pitch, startTicks, duration, note.velocity);
  }
  
  return Buffer.from(file.toBytes());
}

function getKeyNumber(key, mode = 'major') {
  const keyNumbers = {
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
    'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7
  };
  
  let keyNumber = keyNumbers[key] || 0;
  
  // Adjust for minor keys (subtract 3 semitones)
  if (mode === 'minor') {
    keyNumber -= 3;
  }
  
  return keyNumber;
}

module.exports = {
  convertToMidi,
  parseTokens,
  createMidiFile
};