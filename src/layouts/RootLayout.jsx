import React from "react";
import { Outlet } from "react-router-dom";
import { ClerkProvider } from "@clerk/clerk-react";

import Navbar from "../components/Navbar";
const Dashboard = () => {
  const PUBLISHABLE_KEY = "pk_test_Y29taWMtaGFyZS0xLmNsZXJrLmFjY291bnRzLmRldiQ";

  if (!PUBLISHABLE_KEY) {
    throw new Error("Missing Publishable Key");
  }

  return (
    <ClerkProvider
      publishableKey={PUBLISHABLE_KEY}
      signInFallbackRedirectUrl="/dashboard"
    >
      <Navbar />
      <main>
        <Outlet />
      </main>
    </ClerkProvider>
  );
};

export default Dashboard;
