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

// // Sample data
const user = {
  name: "Tejas Srikanth",
  avatar: "https://via.placeholder.com/150", // Replace with actual image URL
  description:
    "Software Developer based in San Francisco. Passionate about technology and open-source.",
  email: "tejas.srikanth@example.com",
  phone: "+1-234-567-8900",
  medicalConditions: [
    "Alzheimer's disease - affects memory and cognitive functions.",
    "High blood pressure - requires regular monitoring and medication.",
    "Heart condition - needs medication and periodic check-ups.",
    "Type II Diabetes - needs Insulin",
    "See more.",
  ],
};

const exerciseData = [
  {
    title: "Jumping Jacks",
    image: `${process.env.PUBLIC_URL}/jumping.webp`,
    description:
      "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
  },
  {
    title: "Lunges",
    image: `${process.env.PUBLIC_URL}/lunge.webp`,
    description:
      "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
  },
  {
    title: "Pushups",
    image: `${process.env.PUBLIC_URL}/pushup.webp`,
    description:
      "A relaxing vacation on the beautiful beaches of Bali, enjoying the sun and surf.",
  },
];


const Fitness = () => {
  return (
    <Box
      component="main"
      sx={{ flexGrow: 1, bgcolor: "background.default", mr: "10px", p: 3 }}
    >
      
      <Typography variant="h4" sx={{ my: 4, fontWeight: "bold" }} gutterBottom>
        Fitness
      </Typography>
      <Typography variant="h6" sx={{ my: 4, fontWeight: "light" }} gutterBottom>
        Choose a workout to learn.
      </Typography>
      <Grid container spacing={4}>
        {exerciseData.slice(0, 3).map((memory, index) => (
          <Grid item xs={12} sm={11} md={4} key={index}>
            <Card 
              variant="outlined"
              sx={{
                maxWidth: "600px",
              }}
              >
              <CardMedia
                component="img"
                fullHeight
                fullWidth
                image={memory.image}
                alt={memory.title}
              />
              <CardContent>
                <Typography variant="h6" component="div">
                  {memory.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {memory.description}
                </Typography>
                <Button
                  variant="outlined"
                  style={{marginTop: "5px"}}
                  sx={{
                    p: 1,
                    borderColor: "#fcd34d",
                    borderWidth: "2px",
                    color: "#000000",
                    fontWeight: "bold",
                    textTransform: "none",
                    "&:hover": {
                      backgroundColor: "#fcd34d",
                    },
                  }}
                  disableElevation
                  component={Link}
                  to="/#0"
                  size="medium"
                >
                  Start
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 2 }}>
        <Button
          variant="outlined"
          fullWidth
          sx={{
            p: 1,
            borderColor: "#fcd34d",
            borderWidth: "2px",
            color: "#000000",
            fontWeight: "bold",
            textTransform: "none",
            "&:hover": {
              backgroundColor: "#fcd34d",
            },
          }}
          disableElevation
          component={Link}
          to="/dashboard"
          size="medium"
        >
          View more memories
        </Button>
      </Box>
    </Box>
  );
};

export default Fitness;
