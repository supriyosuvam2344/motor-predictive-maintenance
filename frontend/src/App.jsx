import { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const getUrgencyColor = (urgencyText) => {
  if (!urgencyText) return '#ffffff'; 
  if (urgencyText.includes("CRITICAL")) return '#ff2a2a'; 
  if (urgencyText.includes("WARNING")) return '#ffd700';  
  return '#00fa9a';                                       
};

// --- LIVE TIME TRACKING ---
const LiveCountdown = ({ rulHoursString, timestampString }) => {
  const [timeLeft, setTimeLeft] = useState(0);
  const [isDestroyed, setIsDestroyed] = useState(false);

  useEffect(() => {
    if (!rulHoursString || !timestampString) return;

    // 1. DATE CLEANER: Fix Excel formats so Javascript can read them
    let cleanTime = timestampString;
    if (cleanTime.includes('-')) {
      const datePart = cleanTime.split(' ')[0].split('-');
      const timePart = cleanTime.split(' ')[1] || "00:00";
      // Convert DD-MM-YYYY to MM/DD/YYYY
      cleanTime = `${datePart[1]}/${datePart[0]}/${datePart[2]} ${timePart}`; 
    }

    // 2. CALCULATE "DATE OF DEATH"
    const startMs = new Date(cleanTime).getTime(); 
    const rulMs = parseFloat(rulHoursString) * 3600 * 1000; 
    const deathDateMs = startMs + rulMs;

    // 3. COMPARE WITH REAL TIME EVERY SECOND
    const timer = setInterval(() => {
      const nowMs = new Date().getTime();
      const diffMs = deathDateMs - nowMs;

      if (diffMs <= 0) {
        setIsDestroyed(true);
        clearInterval(timer);
      } else {
        setIsDestroyed(false);
        setTimeLeft(diffMs / 1000); 
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [rulHoursString, timestampString]);

  // 4. TRIGGER THE DESTROYED WARNING
  if (isDestroyed) {
    return <strong style={{ color: '#ff2a2a', backgroundColor: '#330000', padding: '2px 8px', borderRadius: '4px' }}>FATAL: MOTOR DESTROYED</strong>;
  }

  // 5. FORMAT THE COUNTDOWN
  const days = Math.floor(timeLeft / (3600 * 24));
  const hours = Math.floor((timeLeft % (3600 * 24)) / 3600);
  const minutes = Math.floor((timeLeft % 3600) / 60);
  const seconds = Math.floor(timeLeft % 60);

  return (
    <strong style={{ color: '#ff2a2a', fontFamily: 'monospace', fontSize: '1.4rem', letterSpacing: '2px' }}>
      {days}d {hours}h {minutes}m {seconds}s
    </strong>
  );
};
// --------------------------------------------

// The 13 exact features returned by the Python backend
const allFactors = [
  { key: 'Voltage_V', color: '#ffd700' },
  { key: 'Current_A', color: '#00fa9a' },
  { key: 'Power_kW', color: '#32cd32' },
  { key: 'Power_Factor', color: '#20b2aa' },
  { key: 'RPM', color: '#1e90ff' },
  { key: 'Load_Percent', color: '#da70d6' },
  { key: 'Vibration_X_mm_s', color: '#9400d3' },
  { key: 'Vibration_Y_mm_s', color: '#8a2be2' },
  { key: 'Vibration_Z_mm_s', color: '#ba55d3' },
  { key: 'vibration_rms', color: '#ff007f' },
  { key: 'Bearing_Temp_C', color: '#ff4500' },
  { key: 'Temperature_C', color: '#00d2ff' },
  { key: 'Ambient_Temp_C', color: '#f4a460' },
  { key: 'rul_hours', color: '#ffffff' },
  { key: 'WINDING_TEMP', color: '#ff004f' },
  { key: 'TEMP_RISE', color: '#ff7b00' },
  { key: 'EFFICIENCY', color: '#00ffcc' },
  { key: 'CURRENT_SPIKE', color: '#ff2a2a' }
];

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
      // POST the massive CSV to main.py
      const response = await axios.post('https://motor-predictive-maintenance.onrender.com/analyze_dataset', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      console.log("BACKEND DATA:", response.data.motor_data);
      setServerData(response.data.motor_data);
      setActiveUnit(response.data.available_motors[0]); // Auto-select the first motor
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
      
      {/* HEADER & UPLOADER */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', paddingBottom: '20px', borderBottom: '1px solid #333' }}>
        <h1 style={{ margin: 0, color: '#ffffff', letterSpacing: '1px' }}>
          MOTOR HEALTH ANALYZER
        </h1>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <input 
            type="file" 
            accept=".csv, .xlsx" 
            onChange={handleFileChange} 
            style={{ color: '#fff' }}
          />
          <button 
            onClick={uploadAndAnalyze}
            disabled={isUploading}
            style={{ padding: '12px 24px', backgroundColor: isUploading ? '#555' : '#00d2ff', color: '#000', border: 'none', borderRadius: '4px', cursor: isUploading ? 'not-allowed' : 'pointer', fontWeight: 'bold', textTransform: 'uppercase' }}
          >
            {isUploading ? 'Analyzing Dataset...' : 'Run Diagnostics'}
          </button>
        </div>
      </div>

      {/* RENDER DASHBOARD ONLY IF DATA EXISTS */}
      {serverData && activeMotorReport && (
        <>
          {/* MOTOR NAVIGATION TABS */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '30px' }}>
            {Object.keys(serverData).map(unitId => (
              <button
                key={unitId}
                onClick={() => setActiveUnit(unitId)}
                style={{ padding: '10px 20px', backgroundColor: activeUnit === unitId ? '#1e1e1e' : 'transparent', color: activeUnit === unitId ? '#00d2ff' : '#888', border: 'none', borderBottom: activeUnit === unitId ? '3px solid #00d2ff' : '3px solid transparent', cursor: 'pointer', fontSize: '16px', fontWeight: 'bold' }}
              >
                {unitId}
              </button>
            ))}
          </div>

          {/* AI SUMMARY DASHBOARD */}
          <div style={{ padding: '20px', backgroundColor: '#1e1e1e', borderLeft: `5px solid ${activeMotorReport.summary.fault_code === 0 ? '#00e676' : '#ff4c4c'}`, borderRadius: '6px', marginBottom: '30px' }}>
            <h2 style={{ margin: '0 0 10px 0', color: activeMotorReport.summary.fault_code === 0 ? '#00e676' : '#ff4c4c' }}>
              STATUS: {activeMotorReport.summary.physical_status.toUpperCase()}
            </h2>
            <p style={{ fontSize: '1.2rem', margin: '0 0 10px 0', color: '#ccc' }}>
              Remaining Useful Life: <LiveCountdown 
                rulHoursString={activeMotorReport.summary.current_rul_hours} 
                timestampString={activeMotorReport.summary.latest_timestamp} 
              />
            </p>
            <p style={{ fontSize: '1.1rem', margin: '0 0 10px 0', color: getUrgencyColor(activeMotorReport.summary.maintenance_urgency), fontWeight: 'bold' }}>
              {activeMotorReport.summary.maintenance_urgency}
            </p>
            <p style={{ margin: '0 0 10px 0', color: '#aaa' }}>{activeMotorReport.lstm_time_series_analysis}</p>
            
            <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
              <strong style={{ color: '#00d2ff' }}>SHAP AI Insight:</strong>
              <ul style={{ margin: '5px 0 0 0', paddingLeft: '20px', color: '#bbb' }}>
                {activeMotorReport.ai_explanations.map((exp, idx) => <li key={idx}>{exp}</li>)}
              </ul>
            </div>
          </div>

          {/* VERTICAL TELEMETRY STRIP (13 GRAPHS) */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            {allFactors.map(factor => (
              <div key={factor.key} style={{ backgroundColor: '#1e1e1e', padding: '20px', borderRadius: '8px', border: '1px solid #333' }}>
                <h3 style={{ margin: '0 0 15px 0', color: factor.color, textTransform: 'uppercase', letterSpacing: '1px', fontSize: '14px' }}>
                  {activeUnit} | {factor.key.replace(/_/g, ' ')}
                </h3>
                
                <div style={{ width: '100%', height: 250 }}>
                  <ResponsiveContainer>
                    <LineChart data={activeMotorReport.graph_data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="time_step" stroke="#888" tick={{ fill: '#888' }} />
                      <YAxis stroke="#888" tick={{ fill: '#888' }} />
                      <Tooltip contentStyle={{ backgroundColor: '#222', borderColor: '#444', color: '#fff' }} />
                      <Line type="monotone" dataKey={factor.key} stroke={factor.color} strokeWidth={2} dot={false} activeDot={{ r: 8 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default App;