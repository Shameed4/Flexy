'use strict';

import React, { useState, useRef, useCallback } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Grid,
} from "@mui/material";
import Webcam from "react-webcam";

const SignInPage = () => {
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [cameraApproved, setCameraApproved] = useState(false);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
    console.log("image captured");
  }, [webcamRef]);

  return (
    <Box
      component="main"
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        backgroundColor: "#f5f5f5",
      }}
    >
      <Card
        variant="outlined"
        sx={{
          width: "400px",
          padding: "30px",
          borderRadius: "12px",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
          backgroundColor: "#ffffff",
        }}
      >
        <Typography
          variant="h5"
          sx={{
            textAlign: "center",
            fontWeight: "bold",
            marginBottom: "20px",
          }}
        >
          Sign In to Flexy
        </Typography>

        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Email"
                variant="outlined"
                type="email"
                required
              />
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Password"
                variant="outlined"
                type="password"
                required
              />
            </Grid>

            <Grid item xs={12}>
              <Button
                fullWidth
                variant="contained"
                style={{ backgroundColor: "#fcd34d", color: "#713f12" }}
                sx={{ textTransform: "none", fontWeight: "bold", mt: 2 }}
              >
                Sign In
              </Button>
              <Button
                fullWidth
                variant="contained"
                style={{ backgroundColor: "#fcd34d", color: "#713f12" }}
                sx={{ textTransform: "none", fontWeight: "bold", mt: 2 }}
                onClick={() => setCameraApproved(!cameraApproved)}
              >
                Open Webcam
              </Button>
            </Grid>

            <Grid item xs={12} sx={{ textAlign: "center", marginTop: "10px" }}>
              <Typography variant="body2">
                Don't have an account?{" "}
                <Button variant="text" color="primary" sx={{ fontWeight: "bold", textTransform: "none" }}>
                  Sign Up
                </Button>
              </Typography>
            </Grid>
          </Grid>
        </CardContent>

        {/* Open Webcam Button */}
        <CardContent sx={{ textAlign: "center", marginTop: "20px" }}>
          {/* <Button
            variant="outlined"
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
            onClick={() => setCameraApproved(!cameraApproved)}
          >
            Open Webcam
          </Button> */}

          {cameraApproved && (
            <div className="webcam-container" style={{ marginTop: "20px" }}>
              <Webcam height={300} width={300} ref={webcamRef} />
              <div className="btn-container" style={{ marginTop: "10px" }}>
                <Button
                  variant="contained"
                  style={{ backgroundColor: "#fcd34d", color: "#713f12" }}
                  color="primary"
                  onClick={capture}

                  sx={{ textTransform: "none", fontWeight: "bold" }}
                >
                  Capture photo
                </Button>
              </div>
              {imgSrc && (
                <div style={{ marginTop: "20px" }}>
                  <Typography variant="body2">Captured Photo:</Typography>
                  <img src={imgSrc} alt="Captured" style={{ width: "100%", marginTop: "10px" }} />
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default SignInPage;
