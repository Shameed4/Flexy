import React from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
} from "@mui/material";

const Stats = ({ user }) => {
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
        {user.stats.slice(0, 3).map((memory, index) => (
          <Grid item xs={10} sm={10} md={4} key={index}>
            <Card
              variant="outlined"
              sx={{
                maxWidth: "600px",
              }}
              style={{ backgroundColor: "#fde68a" }}
            >
              <CardContent>
                <Typography variant="h6" component="div">
                  {memory.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {memory.description}
                </Typography>
                <Typography
                  variant="h1"
                  color="text.secondary"
                  style={{ justifyContent: "center" }}
                >
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
        {user.recentlyCompleted.slice(0, 3).map((memory, index) => (
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
        {user.recommendations.slice(0, 3).map((memory, index) => (
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