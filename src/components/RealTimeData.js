import React, { useState, useEffect } from 'react';
// import { Link } from 'react-router-dom';
import axios from 'axios';
import Plot from 'react-plotly.js';
// import Image from './psd_topomap.png';
import Countdown from './Countdown';

// let Image = require('./psd_topomap.png')

const RealTimeData = () => {
  const [plotData, setData] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleAction = async () => {
    try{
      setIsProcessing(true);
      const response = await axios.get('http://localhost:5000/recordstream');
      console.log(response);
      window.location.href = `/display`;
      
    }
    catch(error){
      console.error('Error fetching data:',error)
    }
  };
  
  useEffect(() => {

    const setStream = async () => {
        try {
          const response = await axios.get('http://localhost:5000/startstream');
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      };
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/fetchstream');
        const newData = response.data.data;
        console.log(isProcessing)
        setData(newData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    
    const intervalId = setInterval(fetchData, 100); // Fetch data every 1 second
    setStream();
    return () => clearInterval(intervalId); // Clean up interval on unmount
  }, []);

  return (
    <div className="container mt-5">
      <div className="card">
        <div className="card-body">
          <h1 className="card-title text-center">Human Sentiment Analysis using Brain Maps</h1>
          {isProcessing ? (
            <div className="text-center">
              <p>Please wait while we process your data...</p>
              <Countdown />
            </div>
          ) : (
            <ul className="list-group list-group-flush">
              <li className="list-group-item">
                <button onClick={handleAction} className="btn btn-primary">Record Data</button>
              </li>
            </ul>
          )}
        </div>
      </div>
      <div className="row mt-4">
        <div className="col-md-8 offset-md-2">
          {plotData.map((plot, index) => (
            <div key={index} className="mb-4">
              {/* <h2 className="text-center">{plot.layout.title}</h2> */}
              <Plot data={plot.data} layout={plot.layout} />
            </div>
          ))}
        </div>
      </div>
      {/* Uncomment this section if you want to add an image */}
      {/* <div className="row mt-4">
        <div className="col">
          <img src={Image} className="img-fluid" alt="Topo-map" />
        </div>
      </div> */}
    </div> 
  );
  
};

export default RealTimeData;
