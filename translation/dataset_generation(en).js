const fs = require('fs');
const readline = require('readline');

// Define the input and output filenames
const inputFile = "english_texts.csv";
const outputFile = "en_hin_translated.csv";

// Transcript column index (assuming 0-based indexing)
const transcriptIndex = 2;

// Function to read the CSV file line by line
const readCSV = async (inputFile, callback) => {
  const rl = readline.createInterface({
    input: fs.createReadStream(inputFile),
    crlfDelay: Infinity
  });

  let headerRow = null;

  for await (const line of rl) {
    const data = line.split(',');

    if (!headerRow) {
      // Check for header row and store column index
      headerRow = data;
      if (headerRow[transcriptIndex] !== 'transcript') {
        console.error("Error: 'transcript' column not found at index", transcriptIndex);
        return;
      }
      continue;
    }

    callback(data[transcriptIndex]);
  }

  rl.on('close', () => console.log("Finished reading", inputFile));
};

// Function to write data to the output file
const writeToFile = (data, outputFile) => {
  const writeStream = fs.createWriteStream(outputFile, { flags: 'a' });
  writeStream.write(data + '\n');
  writeStream.on('finish', () => console.log("Data appended to", outputFile));
  writeStream.on('error', err => console.error("Error writing to file:", err));
  writeStream.close();
};

// Read transcript data and write to output file
readCSV(inputFile, (transcript) => {
  writeToFile(transcript, outputFile);
});