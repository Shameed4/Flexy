import React, { useState } from "react";
import { Box, Typography, Grid, Card, CardContent, CardMedia, Button, Rating } from "@mui/material";

const exerciseData = [
  {
    title: "Wrist",
    image: `${process.env.PUBLIC_URL}/hand.webp`,
    description: "Exercise to improve wrist mobility and increase flexibility.",
  },
  {
    title: "Neck",
    image: `${process.env.PUBLIC_URL}/neck.webp`,
    description: "Exercise to enhance neck posture and reduce muscle tension.",
  },
  {
    title: "Shoulder",
    image: `${process.env.PUBLIC_URL}/shoulder.webp`,
    description: "Shoulder exercises to improve mobility and reduce shoulder pain.",
  },
];

const Profile = () => {
  const [ratingValue, setRatingValue] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSecondExercise, setIsSecondExercise] = useState(false);
  const [lastExercise, setLastExercise] = useState('first'); // Track the last exercise

  // Function to calculate stars based on the returned value (0-100)
  const calculateRating = (value) => {
    console.log("Calculating rating for value:", value);
    if (value <= 50) return 0;
    const rating = Math.min(Math.floor((value - 50) / 10) + 1, 5);
    console.log("Calculated rating:", rating);
    return rating;
  };

  // Handle button click and simulate the GET request for the current exercise
  const handleStart = async () => {
    setIsLoading(true);
    setLastExercise('first'); // Track that the first exercise is being performed

    console.log("Starting first exercise...");
    fetch('http://127.0.0.1:5000/stretch-circle', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Response from first exercise API:", data);
        const value = parseFloat(data.output);
        console.log("Parsed output value for first exercise:", value);
        setRatingValue(calculateRating(value));
        setIsLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching data for first exercise:', error);
        setIsLoading(false);
      });
  };

  // Handle the "Redo" button to repeat the latest exercise
  const handleRedo = async () => {
    console.log("Redoing the last exercise:", lastExercise);
    if (lastExercise === 'first') {
      handleStart();  // Redo the first exercise
    } else if (lastExercise === 'second') {
      handleNextExercise();  // Redo the second exercise
    }
  };

  // Handle the "Next Exercise" button to make a different request
  const handleNextExercise = async () => {
    setIsLoading(true);
    setLastExercise('second'); // Track that the second exercise is being performed

    console.log("Starting second exercise...");
    fetch('http://127.0.0.1:5000/stretch-loop', {  // Different endpoint for next exercise
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Response from second exercise API:", data);
        const value = parseFloat(data.output);
        console.log("Parsed output value for second exercise:", value);
        setRatingValue(calculateRating(value));
        setIsLoading(false);
        setIsSecondExercise(true); // Mark that second workout is done
      })
      .catch((error) => {
        console.error('Error fetching data for second exercise:', error);
        setIsLoading(false);
      });
  };

  // Handle "Back to Exercises" button to reset the state and go back to the cards
  const handleBackToExercises = () => {
    console.log("Going back to exercise cards");
    setRatingValue(null);  // Reset the rating
    setIsSecondExercise(false);  // Reset second workout state
    setLastExercise('first');  // Reset to first exercise by default
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
    console.log("Rendering rating:", ratingValue);
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
          onClick={handleRedo}
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

        {/* Next Exercise Button, only shown if rating is greater than or equal to 4 */}
        {ratingValue >= 4 && !isSecondExercise && (
          <Button
            variant="outlined"
            onClick={handleNextExercise}
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
            Next Exercise
          </Button>
        )}

        {isSecondExercise && ratingValue >= 4 && (
          <Button
            variant="outlined"
            onClick={handleBackToExercises}
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
        )}
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
        Stretches
      </Typography>
      <Typography variant="h6" sx={{ my: 4, fontWeight: "light" }} gutterBottom>
        Choose an area of the body to stretch.
      </Typography>
      <Grid container spacing={4}>
        {exerciseData.map((memory, index) => (
          <Grid item xs={12} sm={11} md={4} key={index}>
            <Card variant="outlined" sx={{ maxWidth: "600px" }}>
              <CardMedia
                component="img"
                sx={{
                  height: "350px",
                  width: "100%",
                  objectFit: "cover",
                }}
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
                  onClick={handleStart}
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

export default Profile;
