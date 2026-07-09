import React, { useState, useEffect } from 'react';

const getUrgencyColor = (urgencyText) => {
  if (!urgencyText) return '#ffffff'; 
  if (urgencyText.includes("CRITICAL")) return '#ff2a2a'; 
  if (urgencyText.includes("WARNING")) return '#ffd700';  
  return '#00fa9a';                                       
};

const LiveCountdown = ({ rulHoursString, timestampString }) => {
  const [timeLeft, setTimeLeft] = useState(0);
  const [isDestroyed, setIsDestroyed] = useState(false);

  useEffect(() => {
    if (!rulHoursString || !timestampString) return;
    let cleanTime = timestampString;
    if (cleanTime.includes('-')) {
      const datePart = cleanTime.split(' ')[0].split('-');
      const timePart = cleanTime.split(' ')[1] || "00:00";
      cleanTime = `${datePart[1]}/${datePart[0]}/${datePart[2]} ${timePart}`; 
    }
    const startMs = new Date(cleanTime).getTime(); 
    const rulMs = parseFloat(rulHoursString) * 3600 * 1000; 
    const deathDateMs = startMs + rulMs;

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

  if (isDestroyed) {
    return <strong style={{ color: '#ff2a2a', backgroundColor: '#330000', padding: '2px 8px', borderRadius: '4px' }}>FATAL: MOTOR DESTROYED</strong>;
  }
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

const StatusSummary = ({ report }) => {
  if (!report) return null;
  return (
    <div style={{ padding: '20px', backgroundColor: '#1e1e1e', borderLeft: `5px solid ${report.summary.fault_code === 0 ? '#00e676' : '#ff4c4c'}`, borderRadius: '6px', marginBottom: '30px' }}>
      <h2 style={{ margin: '0 0 10px 0', color: report.summary.fault_code === 0 ? '#00e676' : '#ff4c4c' }}>
        STATUS: {report.summary.physical_status.toUpperCase()}
      </h2>
      <p style={{ fontSize: '1.2rem', margin: '0 0 10px 0', color: '#ccc' }}>
        Remaining Useful Life: <LiveCountdown rulHoursString={report.summary.current_rul_hours} timestampString={report.summary.latest_timestamp} />
      </p>
      <p style={{ fontSize: '1.1rem', margin: '0 0 10px 0', color: getUrgencyColor(report.summary.maintenance_urgency), fontWeight: 'bold' }}>
        {report.summary.maintenance_urgency}
      </p>
      <p style={{ margin: '0 0 10px 0', color: '#aaa' }}>{report.lstm_time_series_analysis}</p>
      
      <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
        <strong style={{ color: '#00d2ff' }}>SHAP AI Insight:</strong>
        <ul style={{ margin: '5px 0 0 0', paddingLeft: '20px', color: '#bbb' }}>
          {report.ai_explanations.map((exp, idx) => <li key={idx}>{exp}</li>)}
        </ul>
      </div>
    </div>
  );
};

export default StatusSummary;