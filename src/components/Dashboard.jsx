import React, { useState } from "react";
import Searchbar from "./Searchbar";
import Sidebar from "./Sidebar";
import Profile from "./Profile";
import Stats from "./Stats";
import Fitness from "./Fitness";
import ChatApp from "./ChatApp";
import { Box } from "@mui/material";

const Dashboard = () => {
  // State to hold user data
  const [user, setUser] = useState(null);
  const [mainArea, setMainArea] = useState("");

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      <Sidebar setMainArea={setMainArea} user={user} />
      <Box
        component="main"
        sx={{ flexGrow: 1, bgcolor: "background.default", p: 3 }}
      >
        {mainArea == "Stats" || mainArea == "" ? ( // physical therapy
          <Stats user={user} />
        ) : mainArea == "Stretches" ? ( //other workout reccomendations
          <Profile user={user} />
        ) : mainArea == "Fitness" ? ( //exercises (sean)
          <Fitness user={user} />
        ) : mainArea == "ChatBot" ? (
          <ChatApp user={user} />
        ) : (
          <Profile user={user} />
        )}
      </Box>
    </Box>
  );
};

export default Dashboard;