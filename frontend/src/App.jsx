import { useState } from 'react';
import axios from 'axios';
import Uploader from './components/Uploader';
import StatusSummary from './components/StatusSummary';
import Dashboard from './components/Dashboard';
import Loader from './components/Loader';

function App() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [serverData, setServerData] = useState(null);
  const [activeUnit, setActiveUnit] = useState(null);
  
  // New state to control the visibility of the popup overlay
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadAndAnalyze = async () => {
    if (!file) return alert("Please select a CSV file first!");
    
    setIsModalOpen(false); // Instantly close the popup if it's open
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post('https://mitaliswarooop-motor-predictive-api.hf.space/analyze_dataset', formData);
      setServerData(response.data.motor_data);
      setActiveUnit(response.data.available_motors[0]); 
    } catch (error) {
      console.error("Backend Error:", error);
      alert(error.response?.data?.detail || "An error occurred connecting to the FastAPI server.");
    } finally {
      setIsUploading(false);
    }
  };

  const activeMotorReport = serverData && activeUnit ? serverData[activeUnit] : null;

  return (
    <div style={{ padding: '40px', fontFamily: 'system-ui, sans-serif', backgroundColor: '#121212', color: '#e0e0e0', minHeight: '100vh', position: 'relative' }}>
      
      {/* 1. Header with Flexbox to push the title left and button right */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '20px', borderBottom: '1px solid #333', marginBottom: '50px' }}>
        <h1 style={{ margin: 0, color: '#ffffff', letterSpacing: '1px' }}>
          MOTOR PREDICTIVE MAINTENANCE
        </h1>
        
        {/* The new "Upload Another File" button (Only shows when results are on screen) */}
        {serverData && !isUploading && (
          <button 
            onClick={() => setIsModalOpen(true)}
            style={{
              padding: '10px 20px', 
              backgroundColor: 'transparent', 
              color: '#00d2ff', 
              border: '1px solid #00d2ff', 
              borderRadius: '4px', 
              cursor: 'pointer', 
              fontWeight: 'bold',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={(e) => { e.target.style.backgroundColor = 'rgba(0, 210, 255, 0.1)'; }}
            onMouseOut={(e) => { e.target.style.backgroundColor = 'transparent'; }}
          >
            + Upload Another File
          </button>
        )}
      </div>

      {/* 2. Show the Uploader normally on the very first visit */}
      {!isUploading && !serverData && (
        <Uploader 
          handleFileChange={handleFileChange} 
          uploadAndAnalyze={uploadAndAnalyze} 
          isUploading={isUploading} 
        />
      )}

      {/* 3. The massive loader sequence */}
      {isUploading && <Loader />}

      {/* 4. The Dashboard Results */}
      {!isUploading && serverData && activeMotorReport && (
        <>
          <StatusSummary report={activeMotorReport} />
          <Dashboard 
            serverData={serverData} 
            activeUnit={activeUnit} 
            setActiveUnit={setActiveUnit} 
            report={activeMotorReport} 
          />
        </>
      )}

      {/* 5. THE MODAL OVERLAY (Only renders when isModalOpen is true) */}
      {isModalOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          backgroundColor: 'rgba(0, 0, 0, 0.75)', // Darkens the background
          backdropFilter: 'blur(8px)', // Creates the hazy/blurry effect
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000 // Ensures it sits on top of everything else
        }}>
          {/* A wrapper to hold the uploader and a close button */}
          <div style={{ position: 'relative', width: '100%', maxWidth: '700px' }}>
            
            {/* An X button to cancel and close the popup */}
            <button 
              onClick={() => setIsModalOpen(false)} 
              style={{ 
                position: 'absolute', 
                top: '-40px', 
                right: '10px', 
                background: 'none', 
                border: 'none', 
                color: '#aaa', 
                fontSize: '28px', 
                cursor: 'pointer' 
              }}
              title="Close"
            >
              &times;
            </button>
            
            {/* We just recycle your exact same Uploader component! */}
            <Uploader 
              handleFileChange={handleFileChange} 
              uploadAndAnalyze={uploadAndAnalyze} 
              isUploading={isUploading} 
            />
          </div>
        </div>
      )}
      
    </div>
  );
}

export default App;