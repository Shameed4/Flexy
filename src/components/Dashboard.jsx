import React, { useState } from "react";
import Searchbar from "./Searchbar";
import Sidebar from "./Sidebar";
import Profile from "./Profile";
import Stats from "./Stats";
import Fitness from "./Fitness";
import ChatApp from "./ChatApp";
import { Box } from "@mui/material";

const Dashboard = () => {
  // Initialize the user data
  const [user, setUser] = useState({
    stats: [
      {
        title: "Daily Streak",
        description: "Flexy every day!",
        number: 5,
      },
      {
        title: "Overall Accuracy",
        description: "Not flexy, you know it.",
        number: 2,
      },
      {
        title: "Exercises Done",
        description: "Super flexy!",
        number: 68,
      },
    ],
    recentlyCompleted: [],
    recommendations: [
      {
        title: "Legs",
        description:
          "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
      },
      {
        title: "Shoulder",
        description:
          "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
      },
      {
        title: "Ankles",
        description:
          "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
      },
    ],
    recentlyPlayed: [],
    accuracyIncrements: 0,
    totalExercises: 0,
  });

  const [mainArea, setMainArea] = useState("");

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      <Sidebar setMainArea={setMainArea} user={user} />
      <Box
        component="main"
        sx={{ flexGrow: 1, bgcolor: "background.default", p: 3 }}
      >
        {mainArea == "Stats" || mainArea == "" ? ( // physical therapy
          <Stats user={user} setUser={setUser} />
        ) : mainArea == "Stretches" ? ( //other workout reccomendations
          <Profile user={user} setUser={setUser} />
        ) : mainArea == "Fitness" ? ( //exercises (sean)
          <Fitness user={user} setUser={setUser} />
        ) : mainArea == "ChatBot" ? (
          <ChatApp user={user} setUser={setUser} />
        ) : (
          <Profile user={user} setUser={setUser} />
        )}
      </Box>
    </Box>
  );
};

export default Dashboard;