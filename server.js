require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const modelBridge = require('./utils/modelBridge');
const omrConverter = require('./utils/omrConverter');
const { spawn } = require('child_process');

// Load vocabulary and create reverse mapping
const vocab = require('./vocab.json');
const reverseVocab = {};
for (const [key, value] of Object.entries(vocab)) {
  reverseVocab[value] = key;
}

// Token to semantic text mapping
function convertTokensToSemantic(tokens) {
  const semanticTokens = tokens.map(token => reverseVocab[token]).filter(token => token);
  
  return semanticTokens.join(' ')
}

// Python image preprocessing function
async function preprocessImagePython(imagePath) {
  return new Promise((resolve, reject) => {
    const outputJsonPath = path.join(__dirname, 'uploads', `preprocess_${Date.now()}.json`);
    const pythonScript = path.join(__dirname, 'python/preprocess.py');
    
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
        const resultData = await fs.promises.readFile(outputJsonPath, 'utf8');
        const result = JSON.parse(resultData);
        
        await fs.promises.unlink(outputJsonPath).catch(() => {});
        
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

const app = express();
const PORT = process.env.PORT || 3000;

const upload = multer({
  dest: 'uploads/',
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|bmp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

app.use(cors());
app.use(express.json());

app.post('/convert', upload.single('image'), async (req, res) => {
  let tempFilePath = null;
  let modelOutputFile = null;
  let midiOutputFile = null;
  
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    tempFilePath = req.file.path;
    
    // Preprocess image using Python script
    const processedImage = await preprocessImagePython(tempFilePath);
    
    const modelOutput = await modelBridge.predict(processedImage);
    
    // Save model output to temporary file in semantic format
    modelOutputFile = path.join(__dirname, 'uploads', `model_output_${Date.now()}.txt`);
    const semanticText = convertTokensToSemantic(modelOutput.tokens);
    await fs.promises.writeFile(modelOutputFile, semanticText);
    
    // Convert using OMR-3.0
    midiOutputFile = await omrConverter.convertWithOMR(modelOutputFile);
    
    // Send MIDI file back to client
    res.setHeader('Content-Type', 'audio/midi');
    res.setHeader('Content-Disposition', 'attachment; filename="converted.mid"');
    
    const midiBuffer = await fs.promises.readFile(midiOutputFile);
    res.send(midiBuffer);
    
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({ 
      error: 'Failed to process image',
      details: error.message 
    });
  } finally {
    // Clean up temporary files (keep model output and image for debugging)
    const filesToClean = [midiOutputFile].filter(Boolean);
    for (const file of filesToClean) {
      try {
        if (fs.existsSync(file)) {
          await fs.promises.unlink(file);
        }
      } catch (cleanupError) {
        console.warn('Failed to cleanup file:', file, cleanupError.message);
      }
    }
    
    // Log the model output for debugging
    if (modelOutputFile && fs.existsSync(modelOutputFile)) {
      try {
        const content = await fs.promises.readFile(modelOutputFile, 'utf8');
        console.log('Generated semantic text:', content);
      } catch (error) {
        console.warn('Failed to read model output for debugging:', error.message);
      }
    }
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ error: 'File too large' });
    }
  }
  
  if (error.message === 'Only image files are allowed') {
    return res.status(400).json({ error: error.message });
  }
  
  console.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});