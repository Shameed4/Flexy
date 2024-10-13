// src/Login.js

import React, { useState } from 'react';
import axios from 'axios';

function Login() {
    const [username, setUsername] = useState('');
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file || !username) {
            setMessage('Please provide a username and a face image.');
            return;
        }

        const formData = new FormData();
        formData.append('username', username);
        formData.append('file', file);

        try {
            const response = await axios.post('/login', formData);
            setMessage(response.data.message);
        } catch (error) {
            setMessage('Login failed');
        }
    };

    return (
        <div>
            <h2>Login</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Username:</label>
                    <input type="text" value={username} onChange={e => setUsername(e.target.value)} required />
                </div>
                <div>
                    <label>Face Image:</label>
                    <input type="file" accept="image/*" onChange={e => setFile(e.target.files[0])} required />
                </div>
                <button type="submit">Login</button>
            </form>
            {message && <p>{message}</p>}
        </div>
    );
}

export default Login;
