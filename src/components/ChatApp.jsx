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
import OpenAI from "openai/index.mjs";


// const openai = new OpenAI({
//   apiKey: process.env.REACT_APP_OPENAI_API_KEY,
// });

// const fetchCompletion = async (message) => {
//   const completion = await openai.chat.completions.create({
//     model: "gpt-4o-mini",
//     messages: [
//       { role: "system", content: "You are a helpful assistant for a physical therapy / exercise app. Your job is to refer people to any of our exercises that may help them. Keep your response to 2 sentences." },
//       {
//         role: "user",
//         content: ${message},
//       },
//     ],
//   });

//   console.log(completion.choices[0].message);
// };

// fetchCompletion();

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

  // Function to handle form submission (sending message)
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

    // Simulate bot reply
    const botMessage = {
      id: uuidv4(),
      user: "Bot",
      message: `You said: "${inputMessage}"`, // You can replace this with API logic
    };

    // Add bot message after user message
    setTimeout(() => {
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    }, 1000); // Delay bot reply for a more natural feel

    setInputMessage(""); // Clear input field
  };

  // Handle key press to submit the form
  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
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
