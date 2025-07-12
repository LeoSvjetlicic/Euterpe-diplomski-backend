const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

async function preprocessImage(imagePath) {
  return new Promise((resolve, reject) => {
    const outputJsonPath = path.join(__dirname, '../uploads', `preprocess_${Date.now()}.json`);
    const pythonScript = path.join(__dirname, '../python/preprocess.py');
    
    const pythonPath = process.env.PYTHON_PATH || 'python3';
    const pythonProcess = spawn(pythonPath, [pythonScript, imagePath, outputJsonPath]);
    
    let stderr = '';
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        reject(new Error(`Python preprocessing failed with code ${code}: ${stderr}`));
        return;
      }
      
      try {
        // Read the processed data
        const resultData = await fs.readFile(outputJsonPath, 'utf8');
        const result = JSON.parse(resultData);
        
        // Clean up temp file
        await fs.unlink(outputJsonPath).catch(() => {});
        
        if (!result.success) {
          reject(new Error(`Preprocessing error: ${result.error}`));
          return;
        }
        
        resolve({
          data: result.data,
          width: result.width,
          height: result.height
        });
        
      } catch (error) {
        reject(new Error(`Failed to parse preprocessing result: ${error.message}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python preprocessing: ${error.message}`));
    });
  });
}

module.exports = {
  preprocessImage
};