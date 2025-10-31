# pLTV Prediction App

This project is a practical implementation of the concepts described in the TRKKN and Gunnar Griese papers. It aims to build a predictive Lifetime Value (pLTV) tool using a tech stack that is free to use and does not require a credit card.

## Cloud-Native Approach

This project is designed to be cloud-native, with all components running in the cloud. This ensures that the application is scalable, reliable, and accessible from anywhere.

## Tech Stack

*   **Data Collection:** server-side Google Tag Manager (sGTM)
*   **Database:** Neon (serverless PostgreSQL)
*   **Application Platform & ML:** Render
*   **Language:** Python

## Progress

*   Initialized the project with a basic Flask application.
*   Set up a Neon serverless PostgreSQL database.
*   Created a `customers` table in the database to store event data.
*   Created an `/event` endpoint to receive and store customer event data.