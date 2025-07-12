const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

async function predict(processedImage) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '../python/inference.py');
    const tempInputFile = path.join(__dirname, '../uploads/temp_input.json');
    
    const inputData = {
      data: Array.from(processedImage.data),
      width: processedImage.width,
      height: processedImage.height
    };
    
    fs.writeFile(tempInputFile, JSON.stringify(inputData))
      .then(() => {
        const pythonPath = process.env.PYTHON_PATH || 'python3';
        const pythonProcess = spawn(pythonPath, [pythonScript, tempInputFile]);
        
        let stdout = '';
        let stderr = '';
        
        pythonProcess.stdout.on('data', (data) => {
          stdout += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
          stderr += data.toString();
        });
        
        pythonProcess.on('close', async (code) => {
          
          try {
            await fs.unlink(tempInputFile);
          } catch (error) {
            console.warn('Failed to cleanup temp file:', error.message);
          }
          
          if (code !== 0) {
            reject(new Error(`Python process failed with code ${code}: ${stderr}`));
            return;
          }
          
          try {
            const result = JSON.parse(stdout.trim());
            resolve(result);
          } catch (error) {
            reject(new Error(`Failed to parse Python output: ${error.message}`));
          }
        });
        
        pythonProcess.on('error', (error) => {
          reject(new Error(`Failed to start Python process: ${error.message}`));
        });
      })
      .catch(reject);
  });
}

function decodeCTC(logits, blankToken = 0) {
  const decodedSequence = [];
  let previousToken = null;
  
  for (let timeStep = 0; timeStep < logits.length; timeStep++) {
    const currentLogits = logits[timeStep];
    const predictedToken = argmax(currentLogits);
    
    if (predictedToken !== blankToken && predictedToken !== previousToken) {
      decodedSequence.push(predictedToken);
    }
    
    previousToken = predictedToken;
  }
  
  return decodedSequence;
}

function argmax(array) {
  let maxIndex = 0;
  let maxValue = array[0];
  
  for (let i = 1; i < array.length; i++) {
    if (array[i] > maxValue) {
      maxValue = array[i];
      maxIndex = i;
    }
  }
  
  return maxIndex;
}

module.exports = {
  predict,
  decodeCTC
};