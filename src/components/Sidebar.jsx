import React from "react";
import Drawer from "@mui/material/Drawer";
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
} from "@mui/material";
import Face2OutlinedIcon from "@mui/icons-material/Face2Outlined";
import CameraOutlinedIcon from "@mui/icons-material/CameraOutlined";
import Diversity3OutlinedIcon from "@mui/icons-material/Diversity3Outlined";
import TextsmsOutlinedIcon from "@mui/icons-material/TextsmsOutlined";
import LinkOutlinedIcon from "@mui/icons-material/LinkOutlined";

import AnalyticsIcon from '@mui/icons-material/Analytics'; // Stats
import AccessibilityNewIcon from '@mui/icons-material/AccessibilityNew'; // Stretch
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter'; // Fitness

const Sidebar = ({ setMainArea }) => {
  const drawerWidth = 240;

  const handleClick = (text) => {
    setMainArea(text);
  };

  return (
    <Drawer
      variant="permanent"
      open={true}
      sx={{
        height: "80%",
        width: drawerWidth,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: drawerWidth,
          boxSizing: "border-box",
          backgroundColor: "#fdfcf6",
        },
      }}
    >
      <Toolbar />
      <Box
        sx={{
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          height: "100%",
        }}
      >
        <List>
          {["Stats", "Stretches", "Fitness", "ChatBot"].map(
            (text, index) => (
              <ListItem
                key={text}
                disablePadding
                onClick={() => handleClick(text)}
              >
                <ListItemButton>
                  <ListItemIcon>
                    {index == 0 && (
                      <AnalyticsIcon sx={{ color: "#795030" }} />
                    )}
                    {index == 1 && (
                      <AccessibilityNewIcon sx={{ color: "#795030" }} />
                    )}
                    {index == 2 && (
                      <FitnessCenterIcon sx={{ color: "#795030" }} />
                    )}
                    {index == 3 && (
                      <TextsmsOutlinedIcon sx={{ color: "#795030" }} />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={text}
                    sx={{
                      fontFamily: "Inter, sans-serif",
                      fontWeight: "700",
                      fontSize: "1.25rem",
                    }}
                  />
                </ListItemButton>
              </ListItem>
            )
          )}
        </List>

        <ListItem
          sx={{ marginTop: "auto", background: "#fde68a", opacity: "90%" }}
          disablePadding
        >
          <ListItemButton>
            <ListItemIcon>
              <LinkOutlinedIcon sx={{ color: "black" }} />
            </ListItemIcon>
            <ListItemText
              primary="Demo Video"
              sx={{
                fontFamily: "Inter, sans-serif",
                fontWeight: 700,
                fontSize: "1.25rem",
                color: "black",
              }}
            />
          </ListItemButton>
        </ListItem>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
