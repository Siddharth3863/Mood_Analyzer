import React from 'react';

const ImageComponent = () => {
  
  return (
    <>
      <div className="container">
        <div className="row">
          <div className="col">
            <h2>Topomap over 4 channels</h2>
            <img className="img-fluid" src={require('./psd_topomap.png')} alt="Placeholder Image" />
          </div>
        </div>
        <div className="row mt-5">
          <div className="col">
            <h2>PSD of the EEG signals</h2>
            <img className="img-fluid" src={require('./psd_plot.png')} alt="Placeholder Image" />
          </div>
        </div>
        <div className="row mt-5">
          <div className="col">
            <h2>Bar Graph representation of Frequency Bands</h2>
            <img className="img-fluid" src={require('./bar_graph.png')} alt="Placeholder Image" />
          </div>
        </div>
        <div className="row mt-5">
          <div className="col">
            <h2>Bar Graph representation of Mood Analysis</h2>
            <img className="img-fluid" src={require('./mood analysis.png')} alt="Placeholder Image" />
          </div>
        </div>
      </div>
    </>
  );
};

export default ImageComponent;
