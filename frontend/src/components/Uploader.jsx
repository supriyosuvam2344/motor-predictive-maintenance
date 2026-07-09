import React, { useState, useRef } from 'react';

const Uploader = ({ handleFileChange, uploadAndAnalyze, isUploading }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);

  const fileInputRef = useRef(null);
  const CIRCUMFERENCE = 2 * Math.PI * 16;

  const initiateSequence = (file) => {
    if (isAnimating) return;
    setIsAnimating(true);
    setFileUploaded(false);

    setTimeout(() => {
      setIsAnimating(false);
      setFileUploaded(true);
      const mockEvent = { target: { files: [file] } };
      handleFileChange(mockEvent); 
    }, 1000);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!fileUploaded && !isAnimating) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (isAnimating || fileUploaded) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      initiateSequence(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelectChange = (e) => {
    if (isAnimating || fileUploaded) return;
    if (e.target.files && e.target.files.length > 0) {
      initiateSequence(e.target.files[0]);
    }
  };

  const [isHovered, setIsHovered] = useState(false);
  const [isPressed, setIsPressed] = useState(false);

  return (
    <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
      <style>
        {`
          @keyframes circle-fill {
            0% { stroke-dashoffset: ${CIRCUMFERENCE}; }
            100% { stroke-dashoffset: 0; }
          }
          @keyframes big-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          @keyframes fade-in {
            0% { opacity: 0; transform: scale(0.9); }
            100% { opacity: 1; transform: scale(1); }
          }
        `}
      </style>

      <div 
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{ 
          width: '100%', 
          maxWidth: '700px',
          minHeight: '280px', 
          backgroundColor: '#1e1e1e', 
          border: isDragging ? '2px dashed #00d2ff' : '1px solid #333', 
          borderRadius: '12px', 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center', 
          gap: '25px', 
          padding: '40px', 
          textAlign: 'center', 
          transition: 'all 0.3s ease',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
        }}
      >
        <div style={{ width: '80px', height: '80px', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="80" height="80" viewBox="0 0 40 40">
            <circle cx="20" cy="20" r="16" fill="none" stroke="#333" strokeWidth="2.5" />
          </svg>
          {isAnimating && (
            <svg width="80" height="80" viewBox="0 0 40 40" style={{ position: 'absolute', top: 0, left: 0 }}>
              <circle cx="20" cy="20" r="16" fill="none" stroke="#00d2ff" strokeWidth="2.5" strokeLinecap="round" strokeDasharray={`${CIRCUMFERENCE}`} strokeDashoffset={`${CIRCUMFERENCE}`} style={{ animation: 'circle-fill 1s ease-out forwards' }} transform="rotate(-90 20 20)" />
            </svg>
          )}
          {fileUploaded ? (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" style={{ position: 'absolute', animation: 'fade-in 0.3s ease' }}>
              <path d="M20 6L9 17L4 12" />
            </svg>
          ) : (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" style={{ position: 'absolute', animation: isAnimating ? 'big-spin 1.2s cubic-bezier(0.5, 0.1, 0.4, 0.9) infinite' : 'none', opacity: isAnimating ? 0.3 : 1 }}>
              <path d="M12 19V5M5 12l7-7 7 7"/>
            </svg>
          )}
        </div>

        <div style={{ height: '24px' }}>
          <p style={{ margin: 0, color: '#aaa', fontSize: '1.1rem' }}>
            {isAnimating ? 'Reading dataset...' : fileUploaded ? 'Dataset loaded and ready for analysis' : 'Drag files to upload, or'}
          </p>
        </div>

        {/* 4. The "Ghost" Hover Button */}
        <button 
          onClick={fileUploaded ? uploadAndAnalyze : () => fileInputRef.current.click()}
          disabled={isAnimating || isUploading}
          
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => { setIsHovered(false); setIsPressed(false); }}
          onMouseDown={() => setIsPressed(true)}
          onMouseUp={() => setIsPressed(false)}

          style={{ 
            padding: '12px 30px', // Adjusted slightly to account for the 2px border
            // The Magic Swap: Background becomes transparent, text becomes cyan
            backgroundColor: isHovered && !isAnimating && !isUploading ? 'transparent' : '#00d2ff', 
            color: isHovered && !isAnimating && !isUploading ? '#00d2ff' : '#000', 
            border: '2px solid #00d2ff', // Keeps the shape intact when background vanishes
            
            borderRadius: '8px', 
            cursor: isAnimating || isUploading ? 'wait' : 'pointer', 
            fontWeight: 'bold', 
            textTransform: 'uppercase',
            letterSpacing: '1px',
            minWidth: '220px', 
            transition: 'all 0.15s ease-in-out',
            transform: isPressed ? 'scale(0.95)' : 'scale(1)',
            opacity: isAnimating || isUploading ? 0.5 : 1
          }}
        >
          {isAnimating ? 'Uploading...' : fileUploaded ? 'Run Diagnostics' : 'Choose File'}
        </button>
        
        <input type="file" accept=".csv, .xlsx" ref={fileInputRef} onChange={handleFileSelectChange} style={{ display: 'none' }} />
      </div>
    </div>
  );
};

export default Uploader;