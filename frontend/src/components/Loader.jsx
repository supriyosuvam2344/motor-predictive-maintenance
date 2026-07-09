import React, { useState, useEffect } from 'react';

const Loader = () => {
  const [progress, setProgress] = useState(0);
  const [messageIndex, setMessageIndex] = useState(0);

  // 1. Array of dynamic loading messages
  const loadingMessages = [
    "Initializing AI models...",
    "Parsing raw CSV telemetry...",
    "Running LSTM anomaly detection...",
    "Calculating Remaining Useful Life (RUL)...",
    "Generating SHAP explainability insights...",
    "Analysing motor health...",
    "Finalizing dashboard report..."
  ];

  // 2. The Progress Bar Fake-out Logic
  useEffect(() => {
    const progressInterval = setInterval(() => {
      setProgress(oldProgress => {
        if (oldProgress >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return Math.min(oldProgress + (Math.random() * 10), 90);
      });
    }, 400);

    return () => clearInterval(progressInterval);
  }, []);

  // 3. The Message Cycling Logic (Changes every 2.5 seconds)
  useEffect(() => {
    const messageInterval = setInterval(() => {
      setMessageIndex(prevIndex => 
        // Loops back to the first message if it reaches the end
        prevIndex >= loadingMessages.length - 1 ? prevIndex : prevIndex + 1
      );
    }, 2500); 

    return () => clearInterval(messageInterval);
  }, []);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '60px 40px',
      backgroundColor: '#1e1e1e',
      borderRadius: '12px',
      border: '1px solid #333',
      margin: '40px auto',
      maxWidth: '600px',
      textAlign: 'center',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
    }}>
      
      <style>
        {`
          @keyframes big-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          @keyframes fade-in {
            0% { opacity: 0; transform: translateY(5px); }
            100% { opacity: 1; transform: translateY(0); }
          }
          .progress-fill {
            transition: width 0.4s ease;
          }
        `}
      </style>

      {/* Large Blue Spinner */}
      <svg 
        width="64" 
        height="64" 
        viewBox="0 0 24 24" 
        fill="none" 
        stroke="#00d2ff" 
        strokeWidth="3" 
        strokeLinecap="round" 
        style={{ animation: 'big-spin 1.2s cubic-bezier(0.5, 0.1, 0.4, 0.9) infinite', marginBottom: '24px' }}
      >
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>

      {/* Main Text */}
      <h2 style={{ margin: '0 0 10px 0', color: '#ffffff', fontSize: '1.6rem', letterSpacing: '0.5px' }}>
        Analyzing Motor Telemetry
      </h2>
      
      {/* Subtitle - Now uses the dynamic array and a fade animation! */}
      <div style={{ height: '30px', marginBottom: '25px' }}>
        <p 
          key={messageIndex} // React re-triggers the animation when this key changes
          style={{ 
            margin: '0', 
            color: '#00d2ff', // Made this blue so it pops!
            fontSize: '1.1rem',
            animation: 'fade-in 0.5s ease forwards' 
          }}
        >
          {loadingMessages[messageIndex]}
        </p>
      </div>

      {/* The Progress Bar */}
      <div style={{ width: '100%', height: '8px', backgroundColor: '#333', borderRadius: '4px', overflow: 'hidden' }}>
        <div 
          className="progress-fill" 
          style={{ 
            width: `${progress}%`, 
            height: '100%', 
            backgroundColor: '#00d2ff', 
            borderRadius: '4px',
            boxShadow: '0 0 10px #00d2ff'
          }}
        />
      </div>
      
    </div>
  );
};

export default Loader;