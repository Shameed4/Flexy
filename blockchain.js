// blockchain.js

const fs = require('fs');
const path = require('path');

// Simulate blockchain storage using a JSON file
const dataFilePath = path.join(__dirname, 'blockchain_data.json');

const Midnight = require('midnight-sdk');

async function storeOnBlockchain(username, encoding) {
    const encryptedEncoding = encrypt(encoding); // Implement encryption
    const tx = await Midnight.storeData({ username, encryptedEncoding });
    console.log(`Data stored with transaction ID: ${tx.id}`);
}

async function retrieveFromBlockchain(username) {
    const data = await Midnight.retrieveData(username);
    const decryptedEncoding = decrypt(data.encryptedEncoding); // Implement decryption
    console.log(`Retrieved encoding for ${username}: ${decryptedEncoding}`);
}

function storeEncoding(username, encoding) {
    let data = {};
    if (fs.existsSync(dataFilePath)) {
        data = JSON.parse(fs.readFileSync(dataFilePath));
    }

    // Store the encoding under the username
    data[username] = encoding;

    // Write back to the file
    fs.writeFileSync(dataFilePath, JSON.stringify(data));

    console.log('Face encoding stored successfully');
}

function retrieveEncoding(username) {
    if (!fs.existsSync(dataFilePath)) {
        throw new Error('Data file not found');
    }

    const data = JSON.parse(fs.readFileSync(dataFilePath));

    if (!(username in data)) {
        throw new Error('User not found');
    }

    // Return the encoding
    console.log(JSON.stringify(data[username]));
}

// Command line interaction
const args = process.argv.slice(2);
const action = args[0];
const username = args[1];
const encoding = args[2] ? JSON.parse(args[2]) : null;

if (action === 'store' && encoding) {
    storeEncoding(username, encoding);
} else if (action === 'retrieve') {
    retrieveEncoding(username);
} else {
    console.error('Invalid arguments');
    process.exit(1);
}