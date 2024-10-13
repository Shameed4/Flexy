import React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import ArrowForwardIosRoundedIcon from "@mui/icons-material/ArrowForwardIosRounded";
import { Grid } from "@mui/material";
import { Link } from "react-router-dom";

const Landing = () => {
  return (
    <Box
      sx={{
        m: "100px",
        height: "70vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "right",
      }}
    >
      <Grid container alignItems="center">
        <Grid item xs={12} md={6}>
          <Typography
            variant="h1"
            sx={{ textAlign: "left", fontWeight: "bold" }}
          >
            It's flexy and we know it.
          </Typography>
          <Typography
            variant="h5"
            sx={{
              mt: 2,
              fontFamily: "poppins",
              color: "#545454",
            }}
          >
            Helping people with arthritis get more flexible and aid them with pain.
          </Typography>
          <Button
            variant="outlined"
            sx={{
              mt: 4,
              p: 1,
              borderColor: "#e1b44f",
              borderWidth: "2px",
              color: "#000000",
              fontWeight: "bold",
              textTransform: "none",
              width: "30%",
              "&:hover": {
                backgroundColor: "#f89742",
              },
            }}
            endIcon={<ArrowForwardIosRoundedIcon />}
            disableElevation
            component={Link}
            to="/dashboard"
            size="medium"
          >
            Get Started
          </Button>
        </Grid>

        <Grid item xs={12} md={6}>
          <Box
            component="img"
            src={`${process.env.PUBLIC_URL}/assets.jpg`}
            alt="Memory matters"
            sx={{
              width: "90%",
              mx: 13,
              mt: 14,
              borderRadius: "10px",
              padding: "10px"
            }}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Landing;
