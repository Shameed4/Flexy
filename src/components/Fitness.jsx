import React, { useState } from "react";
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
  Rating
} from "@mui/material";

// Sample data
const exerciseData = [
  {
    title: "Jumping Jacks",
    image: `${process.env.PUBLIC_URL}/jumping.webp`,
    description: "Jumping jacks are a full-body workout that improves cardiovascular fitness.",
    workoutType: "fitness-jacks",
  },
  {
    title: "Arm Circles",
    image: `${process.env.PUBLIC_URL}/arms.webp`,
    description: "Arm circles help to strengthen the shoulders and improve range of motion.",
    workoutType: "fitness-circle",
  },
  {
    title: "Knee Raises",
    image: `${process.env.PUBLIC_URL}/legs.webp`,
    description: "Knee raises target the lower abdominal muscles and improve leg strength.",
    workoutType: "fitness-legs",
  },
];

const Fitness = () => {
  const [ratingValue, setRatingValue] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentExercise, setCurrentExercise] = useState(null);

  // Function to calculate stars based on the returned value (0-100)
  const calculateRating = (value) => {
    if (value <= 50) return 0;
    return Math.min(Math.floor((value - 50) / 10) + 1, 5);
  };

  // Function to make the GET request
  const fetchExercise = async (workoutType) => {
    setIsLoading(true);
    fetch(`http://127.0.0.1:5000/${workoutType}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
    })
      .then((response) => response.json())
      .then((data) => {
        setRatingValue("100.00");
        setIsLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching exercise data:', error);
        setIsLoading(false);
      });
  };

  // Handle start button for the first exercise
  const handleStart = (workoutType) => {
    setCurrentExercise(workoutType); // Set the current exercise type
    fetchExercise(workoutType); // Fetch the exercise data
  };

  // Conditionally render the loading message or the rating
  if (isLoading) {
    return (
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
          bgcolor: "background.default",
        }}
      >
        <Typography variant="h4" sx={{ fontWeight: "bold", fontFamily: "Berkshire Swash" }}>
          Wait! Proceeding to Desktop
        </Typography>
      </Box>
    );
  }

  // Conditionally render the rating once the request is successful
  if (ratingValue !== null) {
    return (
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
          bgcolor: "background.default",
        }}
      >
        <Rating
          name="read-only"
          value={ratingValue}
          readOnly
          sx={{ fontSize: "5rem", mb: 3 }}
        />
        <Typography
          variant="h3"
          sx={{
            fontWeight: "bold",
            color: ratingValue >= 4 ? "green" : "red",
          }}
        >
          {ratingValue >= 4 ? "Great job!" : "Try again"}
        </Typography>

        {/* Redo Button */}
        <Button
          variant="outlined"
          onClick={() => fetchExercise(currentExercise)} // Redo the current exercise
          sx={{
            mt: 3,
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
        >
          Redo
        </Button>

        <Button
          variant="outlined"
          onClick={() => setRatingValue(null)} // Reset to show exercises again
          sx={{
            mt: 3,
            p: 1,
            borderColor: "#4caf50",
            borderWidth: "2px",
            color: "#000000",
            fontWeight: "bold",
            textTransform: "none",
            "&:hover": {
              backgroundColor: "#4caf50",
            },
          }}
        >
          Back to Exercises
        </Button>
      </Box>
    );
  }

  // Initial layout with cards before button click
  return (
    <Box
      component="main"
      sx={{ flexGrow: 1, bgcolor: "background.default", mr: "10px", p: 3 }}
    >
      <Typography variant="h4" sx={{ my: 4, fontWeight: "bold" }} gutterBottom>
        Fitness
      </Typography>
      <Typography variant="h6" sx={{ my: 4, fontWeight: "light" }} gutterBottom>
        Choose a workout.
      </Typography>
      <Grid container spacing={4}>
        {exerciseData.map((memory, index) => (
          <Grid item xs={12} sm={11} md={4} key={index}>
            <Card variant="outlined" sx={{ maxWidth: "600px" }}>
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
                  style={{ marginTop: "5px" }}
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
                  onClick={() => handleStart(memory.workoutType)} // Start exercise for this card
                  size="medium"
                >
                  Start
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Fitness;