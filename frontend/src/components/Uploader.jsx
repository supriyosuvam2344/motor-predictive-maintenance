const Uploader = ({ handleFileChange, uploadAndAnalyze, isUploading }) => {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', paddingBottom: '20px', borderBottom: '1px solid #333' }}>
      
      <h1 style={{ margin: 0, color: '#ffffff', letterSpacing: '1px' }}>
        MOTOR PREDICTIVE MAINTENANCE
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
          style={{ 
            padding: '12px 24px', 
            backgroundColor: isUploading ? '#555' : '#00d2ff', 
            color: isUploading ? '#e0e0e0' : '#000', 
            border: 'none', 
            borderRadius: '4px', 
            cursor: isUploading ? 'not-allowed' : 'pointer', 
            fontWeight: 'bold', 
            textTransform: 'uppercase',
            minWidth: '220px', 
            transition: 'all 0.3s ease',
            textAlign: 'center'
          }}
        >
          {isUploading ? 'Analyzing...' : 'Run Diagnostics'}
        </button>
      </div>
    </div>
  );
};

export default Uploader;