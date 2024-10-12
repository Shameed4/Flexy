import React, { useState } from "react";
import { Box, Typography, Grid, Card, CardContent, CardMedia, Button, Rating } from "@mui/material";

const exerciseData = [
  {
    title: "Wrist",
    image: `${process.env.PUBLIC_URL}/wrist.webp`,
    description: "Exercise to improve wrist mobility and increase flexibility.",
    routes: ['stretch-circle', 'stretch-loop'],  // Two routes for wrist exercises
  },
  {
    title: "Neck",
    image: `${process.env.PUBLIC_URL}/neck.webp`,
    description: "Exercise to enhance neck posture and reduce muscle tension.",
    routes: ['neck-oval', 'neck-yesno'],  // Two routes for neck exercises
  },
  {
    title: "Shoulder",
    image: `${process.env.PUBLIC_URL}/shoulder.webp`,
    description: "Shoulder exercises to improve mobility and reduce shoulder pain.",
    routes: ['shoulder-grab', 'shoulder-reach'],  // Two routes for shoulder exercises
  },
];

const Profile = () => {
  const [ratingValue, setRatingValue] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSecondExercise, setIsSecondExercise] = useState(false);
  const [exerciseRoutes, setExerciseRoutes] = useState([]); // Stores the current exercise's routes
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0); // Track the current exercise in a card

  // Function to calculate stars based on the returned value (0-100)
  const calculateRating = (value) => {
    if (value <= 50) return 0;
    return Math.min(Math.floor((value - 50) / 10) + 1, 5);
  };

  // Generalized function to make the GET request
  const fetchExercise = async (route) => {
    setIsLoading(true);
    fetch(`http://127.0.0.1:5000/${route}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
    })
      .then((response) => response.json())
      .then((data) => {
        const value = parseFloat(data.output);
        setRatingValue(calculateRating(value));
        setIsLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching exercise data:', error);
        setIsLoading(false);
      });
  };

  // Handle start button for the first exercise
  const handleStart = (routes) => {
    setExerciseRoutes(routes); // Store the routes for the current exercise
    setCurrentExerciseIndex(0); // Set to the first exercise
    setIsSecondExercise(false); // Reset second exercise flag
    fetchExercise(routes[0]); // Fetch the first exercise
  };

  // Handle next exercise button (fetch the second exercise in the routes)
  const handleNextExercise = () => {
    setIsSecondExercise(true); // Mark that we're doing the second exercise
    setCurrentExerciseIndex(1); // Set to the second exercise
    fetchExercise(exerciseRoutes[1]); // Fetch the second exercise
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
          onClick={() => fetchExercise(exerciseRoutes[currentExerciseIndex])} // Redo the current exercise
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
            onClick={handleNextExercise} // Proceed to next exercise
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
                  onClick={() => handleStart(memory.routes)} // Start exercise for this card
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
