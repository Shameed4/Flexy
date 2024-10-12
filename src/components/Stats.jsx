import React from "react";
import {
  CardHeader,
  Box,
  Toolbar,
  Typography,
  Card,
  CardContent,
  Avatar,
  Grid,
  Divider,
  CardMedia,
  Button,
  Link,
} from "@mui/material";

const recentlyCompleted = [
  {
    title: "Wrist",
    description:
      "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
  },
  {
    title: "Neck",
    description:
      "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
  },
];

const reccomendations = [
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
];

const stats = [
  {
    title: "Daily Streak",
    description:
      "Flexy every day!",
    number: 5
  },
  {
    title: "Overall Accuracy",
    description:
      "Not flexy, you know it.",
    number: 2
  },
  {
    title: "Exercises Done",
    description:
      "Super flexy!",
    number: 68
  },
];


const Stats = () => {
  return (
    <Box
      component="main"
      sx={{ flexGrow: 1, bgcolor: "background.default", mr: "10px", p: 3 }}
    >
      <Typography variant="h4" sx={{ my: 4, fontWeight: "bold" }} gutterBottom>
        Stats
      </Typography>
      <Typography variant="h6" sx={{ my: 4, fontWeight: "light" }} gutterBottom>
        View your progress.
      </Typography>

      <Typography variant="h6" sx={{ my: 4, fontWeight: "bold" }} gutterBottom>
        Stats
      </Typography>
      <Grid container spacing={4}>
        {stats.slice(0, 3).map((memory, index) => (
          <Grid item xs={10} sm={10} md={4} key={index}>
            <Card 
              variant="outlined"
              sx={{
                maxWidth: "600px",
              }}
              style={{backgroundColor: "#fde68a"}}
              >
              <CardContent>
                <Typography variant="h6" component="div">
                  {memory.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {memory.description}
                </Typography>
                <Typography variant="h1" color="text.secondary" style={{justifyContent: "center"}}>
                  {memory.number}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h6" sx={{ my: 4, fontWeight: "bold" }} gutterBottom>
        Recently Completed
      </Typography>
      <Grid container spacing={4}>
        {recentlyCompleted.slice(0, 3).map((memory, index) => (
          <Grid item xs={10} sm={10} md={10} key={index}>
            <Card 
              variant="outlined"
              sx={{
                maxWidth: "600px",
              }}
              >
              <CardContent>
                <Typography variant="h6" component="div">
                  {memory.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {memory.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h6" sx={{ my: 4, fontWeight: "bold" }} gutterBottom>
        Recommended Next Exercises
      </Typography>
      <Grid container spacing={4}>
        {reccomendations.slice(0, 3).map((memory, index) => (
          <Grid item xs={10} sm={10} md={10} key={index}>
            <Card 
              variant="outlined"
              sx={{
                maxWidth: "600px",
              }}
              >
              <CardContent>
                <Typography variant="h6" component="div">
                  {memory.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {memory.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Stats;
