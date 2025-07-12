const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

async function convertWithOMR(modelOutputFile) {
  return new Promise((resolve, reject) => {
    const outputMidiFile = path.join(__dirname, '../uploads', `midi_output_${Date.now()}.mid`);
    
    // Use java directly with the semantic importer
    const omrCommand = 'java';
    const jarPath = path.join(__dirname, 'omr-3.0-SNAPSHOT.jar');
    const omrArgs = [
      '-cp', jarPath,
      'es.ua.dlsi.im3.omr.encoding.semantic.SemanticImporter',
      modelOutputFile,
      outputMidiFile
    ];
    
    const omrProcess = spawn(omrCommand, omrArgs, {
      cwd: path.join(__dirname, '..')
    });
    
    let stderr = '';
    let stdout = '';
    
    omrProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    omrProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    omrProcess.on('close', async (code) => {
      console.log('OMR process output:', stdout);
      console.log('OMR process stderr:', stderr);
      console.log('OMR exit code:', code);
      
      if (code !== 0) {
        reject(new Error(`OMR conversion failed with code ${code}: ${stderr}`));
        return;
      }
      
      try {
        // Verify the output file was created
        await fs.access(outputMidiFile);
        resolve(outputMidiFile);
      } catch (error) {
        reject(new Error(`OMR output file not found: ${outputMidiFile}`));
      }
    });
    
    omrProcess.on('error', (error) => {
      reject(new Error(`Failed to start OMR process: ${error.message}`));
    });
  });
}

module.exports = {
  convertWithOMR
};