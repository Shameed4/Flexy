## Flexy and I know it

## Inspiration

With 528 million people worldwide affected by osteoarthritis and that number expected to surge by 2050, maintaining joint and muscle health is essential. From arthritis and sports injuries to daily joint pain, millions struggle with conditions that limit movement. While there may be no cure, research has shown that targeted stretching and mobility exercises can significantly improve quality of life, reduce pain, and support recovery.

Our app is designed to make rehabilitation more engaging and accessible, helping individuals recover faster and live pain-free. By offering a gamified experience with accurate posture measurement, we ensure users perform exercises correctly, promoting better outcomes and alleviating physical challenges in a fun, user-friendly way.

## What it does

1. We use an advanced posture detection algorithm to identify key body joints and landmarks in real-time videos. This allows the system to map the user's movements with high accuracy and ensure that users are performing therapeutic exercises correctly which is crucial for individuals with arthritis or those recovering from sports injuries. Our custom dataset of various correct exercise postures increases the algorithm's precision making it adaptable to different body types and conditions, ensuring exercises are both safe and effective. This reduces the risk of improper movements which can worsen the condition or slow recovery.

    ![videofitness-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/00cf41a7-3317-46a3-bf24-a1322c22c701)



2. To make rehabilitation engaging, we introduced a gamified feature that allows users to draw shapes like circles, infinity loops, or waves with their body movements. This interactive approach is designed to help both arthritis patients and the general population improve mobility in a fun and rewarding way. By focusing on specific shapes, the system targets key joints—such as the shoulders, arms, and neck—that are commonly affected by arthritis or injuries. Drawing these shapes promotes range-of-motion exercises, which are essential for reducing stiffness and improving flexibility. The gamification aspect keeps users motivated, making it easier to stay consistent with their exercise routines, which is vital for long-term recovery and pain management.

   ![2024-10-1307-13-50-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/f27ada11-704b-4c63-8573-c9a10eeb9af6)


3. We integrated a fine-tuned conversational bot that helps users choose different exercises and movements tailored to their specific needs and goals. The bot provides personalized suggestions based on the user's current mobility, pain levels, and progress, guiding them through a variety of stretches and movements that target specific joints or areas of stiffness.

   ![PNG image](https://github.com/user-attachments/assets/5d8d61f1-6f48-40d0-8c83-04e9203ea033)

## How we built it

Our project begins by capturing real-time video feeds using mobile devices. We utilize MediaPipe and OpenCV to detect key body joints and landmarks within the video, allowing us to map out the user's posture and movements in real-time. We used these tools to create our own custom dataset of demonstration exercises for our users to follow. Computational geometry techniques such as cosine similarity were utilized to determine the similarity between our demonstration and the user's form and it was also used to measure the angles between the landmarks and the drawing. We decided to built a facial authentication system by using blockchain technology and have a front end which has a dashboard of activeness and number of workout session completed. The whole front end was built on React.JS and the backend with the help of Flask. We also took advantage by fine tuning a LLM solely for the purpose of suggesting tailored exercises.


## Challenges we ran into

One major challenge we encountered was the complexity of implementing real-time posture similarity while maintaining low latency across different devices. To ensure the accuracy of our model we had  process live video which was required optimizing our algorithms, especially when detecting subtle movements. We also faced challenges in the integration of computational geometry for our gamified features, such as recognizing body movements to draw precise shapes like waves and circles. Handling variability in user movements while maintaining the integrity of these shapes added a layer of complexity to the project. We also faced problem to properly pre-process our dataset and make the fine tune the cosine similarity algorithm.

## Accomplishments that we're proud of

We are proud of successfully implementing real-time posture detection similarity algorithm that accurately tracks user movements with low latency across different devices. Our integration of computational geometry into the gamified feature allowed us to create an engaging and innovative experience where users can draw shapes like waves and circles to improve mobility. 

## What we learned

We also increased our understanding of computational geometry and how it can be applied to interactive, gamified features. Another key takeaway was the experience of creating your own custom datasets and train the existing model on that.

## What's next for **Flexy and I know it**

We aim to integrate Flexy with VR allowing users to visualize their movements and corrections during exercises in a more immersive way. We're also exploring the use of wearable technology, like smartwatches or sensors, to provide even more precise tracking and feedback on mobility and joint health. 
