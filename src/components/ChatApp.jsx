import React, { useState, useRef } from "react";
import {
  Box,
  TextField,
  InputAdornment,
  Typography,
  Avatar,
  IconButton,
} from "@mui/material";
import SendRoundedIcon from "@mui/icons-material/SendRounded";
import { v4 as uuidv4 } from "uuid";

// Function to send the message to Flask and get the response
async function fetchCompletion(message) {
  try {
    const response = await fetch('http://127.0.0.1:5000/cloudflare_ai', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,  // Send the user message to Flask
      }),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();  // Get the JSON response from Flask
    return data;  // Return the response from Flask
  } catch (error) {
    console.error('Error fetching completion:', error);
    return { error: "Error processing your request" };
  }
}

// Function to generate avatar based on user's name
const stringAvatar = (name) => {
  return {
    sx: {
      bgcolor: "#00796b", // Default color
      color: "#FFF",
    },
    children: `${name[0]}`, // Display first letter
  };
};

// Chatbot component
const Chatbot = () => {
  const [messages, setMessages] = useState([
    { id: uuidv4(), user: "Bot", message: `Hello! How can I assist you today?` },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const formRef = useRef(null);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return; // Prevent empty message

    const userMessage = {
      id: uuidv4(),
      user: "User",
      message: inputMessage,
    };

    // Add user message to the chat
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      // Await the Flask API response
      const completion = await fetchCompletion(userMessage.message);

      // Get the message from the Flask response
      const botMessageContent = completion.result?.response || "Sorry, I couldn't process that.";

      // Create bot message
      const botMessage = {
        id: uuidv4(),
        user: "Bot",
        message: botMessageContent,
      };

      // Add bot message after user message
      setMessages((prevMessages) => [...prevMessages, botMessage]);

    } catch (error) {
      console.error("Error fetching completion:", error);
      const botMessage = {
        id: uuidv4(),
        user: "Bot",
        message: "Sorry, there was an error processing your request.",
      };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    }

    setInputMessage(""); // Clear input field
  };

  // Handle key press to submit the form
  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      formRef.current.requestSubmit();
    }
  };

  return (
    <Box sx={{ padding: "20px", maxWidth: "600px", marginTop: "70px" }}>
      {/* Chat messages display */}
      <Box sx={{ maxHeight: "400px", overflowY: "auto", marginBottom: "16px" }}>
        {messages.map((message) => (
          <Box
            key={message.id}
            sx={{
              display: "flex",
              alignItems: "center",
              marginBottom: "10px",
            }}
          >
            <Avatar {...stringAvatar(message.user)} />
            <Typography
              sx={{
                marginLeft: "10px",
                backgroundColor: "#f5f5f5",
                padding: "10px",
                borderRadius: "10px",
              }}
            >
              <strong>{message.user}:</strong> {message.message}
            </Typography>
          </Box>
        ))}
      </Box>

      {/* Input field for sending messages */}
      <form ref={formRef} onSubmit={handleSendMessage}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type a message..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  onClick={() => formRef.current.requestSubmit()}
                  color="primary"
                >
                  <SendRoundedIcon />
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
      </form>
    </Box>
  );
};

export default Chatbot;
