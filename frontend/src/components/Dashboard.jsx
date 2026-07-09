import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

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

const Dashboard = ({ serverData, activeUnit, setActiveUnit, report }) => {
  if (!serverData || !report) return null;

  return (
    <>
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

      <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
        {allFactors.map(factor => (
          <div key={factor.key} style={{ backgroundColor: '#1e1e1e', padding: '20px', borderRadius: '8px', border: '1px solid #333' }}>
            <h3 style={{ margin: '0 0 15px 0', color: factor.color, textTransform: 'uppercase', letterSpacing: '1px', fontSize: '14px' }}>
              {activeUnit} | {factor.key.replace(/_/g, ' ')}
            </h3>
            
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart 
                  data={report.graph_data} 
                  syncId="motor-metrics"
                  margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="time_step" stroke="#888" tick={{ fill: '#888' }} ticks={[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]} />
                  <YAxis width={60} stroke="#888" tick={{ fill: '#888' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#222', borderColor: '#444', color: '#fff' }} />
                  <Line type="monotone" dataKey={factor.key} stroke={factor.color} strokeWidth={2} dot={false} activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default Dashboard;