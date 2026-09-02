import { initializeApp } from "https://www.gstatic.com/firebasejs/12.18.0/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/12.18.0/firebase-analytics.js";

const firebaseConfig = {
  apiKey: "AIzaSyCs9vCa1syWGYzM8X82ewR3m9HTTMwiCKo",
  authDomain: "sincoai.firebaseapp.com",
  projectId: "sincoai",
  storageBucket: "sincoai.firebasestorage.app",
  messagingSenderId: "1029234881588",
  appId: "1:1029234881588:web:b4dfad591af406507a0661",
  measurementId: "G-N0SQ422HQH",
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export { app, analytics };
