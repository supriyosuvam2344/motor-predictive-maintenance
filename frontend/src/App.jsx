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

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadAndAnalyze = async () => {
    if (!file) return alert("Please select a CSV file first!");
    
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
    <div style={{ padding: '40px', fontFamily: 'system-ui, sans-serif', backgroundColor: '#121212', color: '#e0e0e0', minHeight: '100vh' }}>
      
      <Uploader 
        handleFileChange={handleFileChange} 
        uploadAndAnalyze={uploadAndAnalyze} 
        isUploading={isUploading} 
      />

      {/* Show the massive loader ONLY when uploading */}
      {isUploading && <Loader />}

      {/* Show the Dashboard ONLY when NOT uploading and data exists */}
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
    </div>
  );
}

export default App;